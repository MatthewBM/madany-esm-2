import math
import torch
import logging
import asyncio
import torch.multiprocessing as mp
from abc import ABC, abstractmethod
from transformers import AutoTokenizer, EsmModel, PreTrainedTokenizer
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, List
from config import settings

logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
logger = logging.getLogger(__name__)

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "fp8": getattr(torch, "float8_e4m3fn", torch.float16)
}

class BaseInferenceWorker(ABC):
    """
    Base class for inference workers that defines core methods for api to function
    Further models implementations can build off of this

    :param model_name: ID or model file used to pull weights and tokenizers (such as a HuggingFace model ID)
    :param device_id: Specific GPU index for the worker
    :param is_mock: If true, run on cpu device
    """
    model: Optional[torch.nn.Module]
    tokenizer: Optional[PreTrainedTokenizer]

    def __init__(self, model_name: str, device_id: int, is_mock: bool):
        self.model_name = model_name
        self.device_id = device_id
        self.is_mock = is_mock
        self.device = torch.device("cpu") if is_mock else torch.device(f"cuda:{device_id}")

        self.model = None
        self.tokenizer = None
        self.load()

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def predict(self, sequences: List[str]) -> List[List[float]]:
        pass

class ESM2Worker(BaseInferenceWorker):
    """
    Specific implementation of ESM2 protein language model.
    Handles the transformation of amino acid strings into ESM embeddings.
    """
    def load(self):
        """
        Pull  model and tokenizer from HuggingFace online model repos.
        call eval() immediately to turn off unnecessary training model features.
        pin model to a specific commit for reproducability
        """
        REVISION = "08e4846e537177426273712802403f7ba8261b6c"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, revision=REVISION) # nosec B615
        model = EsmModel.from_pretrained(self.model_name, revision=REVISION) # nosec B615

        self.model = model.to(self.device)
        self.model.eval()
        logger.info(f"[Worker] ESM-2 replica successfully loaded on {self.device}")

    def predict(self, sequences: List[str]) -> List[List[float]]:
        """
        Perform the actual inference on a chunk of protein sequences.
        """
        assert self.tokenizer is not None
        assert self.model is not None

        inputs = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=settings.max_sequence_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        """
        Evaluate where to put tensors and what precision to use.
        Further evaluates whether to employ AMP if in Blackwell context
        """
        device_type = "cuda" if "cuda" in self.device.type else "cpu"
        use_amp = settings.amp_mode != "none"
        active_dtype = settings.amp_mode if use_amp else settings.inference_precision
        infer_dtype = DTYPE_MAP.get(active_dtype, torch.float16)
        out_dtype = DTYPE_MAP.get(settings.output_precision, torch.float32)

        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=infer_dtype, enabled=(device_type == "cuda")):
                outputs = self.model(**inputs)
            """
            Convert raw residue-level hidden states into a single protein- or residue-level representation
            """
            if settings.return_residue_embeddings:
                embeddings = outputs.last_hidden_state.to(out_dtype).cpu().tolist()
            else:
                embeddings = outputs.last_hidden_state.mean(dim=1).to(out_dtype).cpu().tolist()

        return embeddings

class WorkerState:
    """
    Container for model engine in an isolated process.
    Class differs between workers since 'spawn' mp method used.
    """
    engine: Optional[BaseInferenceWorker] = None

def init_worker(device_queue: mp.Queue, is_mock: bool):
    """
    Function called when a new subprocess is spawned.
    Retrieve a GPU ID from the shared queue.
    """
    device_id = device_queue.get()
    model_name = settings.model_name
    WorkerState.engine = ESM2Worker(model_name=model_name, device_id=device_id, is_mock=is_mock)

def process_chunk(sequences: list[str]) -> list[list[float]]:
    """
    Entry point for inference tasks sent from web thread.
    """
    if WorkerState.engine is None:
        raise RuntimeError("Worker engine was not initialized")
    return WorkerState.engine.predict(sequences)

class ESM2Engine:
    """
    Master inference orchestrator that manages all GPU workers.
    Splits large batches of sequences and distributes
    to each available GPU worker.
    """
    def __init__(self):
        self.is_mock = settings.mock_gpu
        self.workers_per_gpu = settings.workers_per_gpu

        if self.is_mock:
            self.gpu_count = settings.mock_gpu_count
            logger.info(f"[MOCK MODE] Simulating {self.gpu_count} GPUs.")
        else:
            self.gpu_count = torch.cuda.device_count()
            logger.info(f"Detected {self.gpu_count} GPUs.")

        self.num_workers = max(1, self.gpu_count) * self.workers_per_gpu
        logger.info(f"Spawning {self.num_workers} total workers.")

        """
        Since inference is split to identical workers on independant GPUs,
        spawn method will clear/partition CUDA context mutually exclusive to each worker.
        """
        self.mp_ctx = mp.get_context("spawn")
        self.manager = self.mp_ctx.Manager()
        self.device_queue = self.manager.Queue()

        for i in range(max(1, self.gpu_count)):
            for _ in range(self.workers_per_gpu):
                self.device_queue.put(i)

        self.executor = ProcessPoolExecutor(
            max_workers=self.num_workers,
            mp_context=self.mp_ctx,
            initializer=init_worker,
            initargs=(self.device_queue, self.is_mock)
        )

    async def predict_single(self, sequence: str) -> list[float]:
        """
        Send and process one sequence to the worker pool and retrieve output
        """
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(self.executor, process_chunk, [sequence])
        return result[0]

    async def predict_batch(self, sequences: list[str]) -> list[list[float]]:
        """
        Take a list of proteins, partition into per-worker chunks,
        and send them to all workers
        """
        total_seqs = len(sequences)

        if self.num_workers == 1:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self.executor, process_chunk, sequences)

        chunk_size = math.ceil(total_seqs / self.num_workers)
        chunks = [sequences[i:i + chunk_size] for i in range(0, total_seqs, chunk_size)]

        loop = asyncio.get_running_loop()
        tasks = []

        for chunk in chunks:
            if not chunk: continue
            task = loop.run_in_executor(self.executor, process_chunk, chunk)
            tasks.append(task)

        chunked_results = await asyncio.gather(*tasks)
        return [emb for chunk in chunked_results for emb in chunk]
