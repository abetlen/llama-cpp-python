from __future__ import annotations

import importlib

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)


CompletionResult = Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]
ChatCompletionResult = Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]
ResponseResult = Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]


def _load_runtime() -> Any:
    try:
        return importlib.import_module("llama_cpp.v1._runtime")
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if not missing.startswith(("openai", "pydantic", "pydantic_core")):
            raise
        raise ImportError(
            "llama_cpp.v1.Model requires the server runtime dependencies. "
            "Install them with `pip install llama-cpp-python[v1]`."
        ) from exc


def _dump_json(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, dict):
        return {key: _dump_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_dump_json(item) for item in value]
    return value


class _CompletionsResource:
    def __init__(self, model: "Model") -> None:
        self._model = model

    def __call__(self, **kwargs: Any) -> CompletionResult:
        return self.create(**kwargs)

    def create(self, **kwargs: Any) -> CompletionResult:
        return self._model.create_completion(**kwargs)


class _ChatCompletionsResource:
    def __init__(self, model: "Model") -> None:
        self._model = model

    def __call__(self, **kwargs: Any) -> ChatCompletionResult:
        return self.create(**kwargs)

    def create(self, **kwargs: Any) -> ChatCompletionResult:
        return self._model.create_chat_completion(**kwargs)


class _ChatResource:
    def __init__(self, completions: _ChatCompletionsResource) -> None:
        self.completions = completions


class _ResponsesResource:
    def __init__(self, model: "Model") -> None:
        self._model = model

    def __call__(self, **kwargs: Any) -> ResponseResult:
        return self.create(**kwargs)

    def create(self, **kwargs: Any) -> ResponseResult:
        return self._model.create_response(**kwargs)


class Model:
    """OpenAI-compatible interface backed by the batch-processing runtime."""

    @dataclass(frozen=True)
    class LoraAdapter:
        path: str
        scale: float = 1.0

    @dataclass(frozen=True)
    class MTMDConfig:
        mmproj_path: str
        embedding_cache_path: Optional[str] = None
        embedding_cache_max_bytes: int = 0
        allowed_media_domains: Optional[List[str]] = None
        allowed_local_media_path: Optional[str] = None
        image_max_bytes: int = 25 * 1024 * 1024
        audio_max_bytes: int = 25 * 1024 * 1024
        image_timeout_seconds: float = 10.0

    @dataclass(frozen=True)
    class DiskCacheConfig:
        path: str
        max_bytes: int
        min_tokens: int = 128

    def __init__(
        self,
        *,
        model_path: str,
        model_alias: Optional[str] = None,
        chat_template: Optional[str] = None,
        loras: Optional[Sequence[Union["Model.LoraAdapter", Dict[str, Any]]]] = None,
        mtmd: Optional[Union["Model.MTMDConfig", Dict[str, Any]]] = None,
        disk_cache: Optional[Union["Model.DiskCacheConfig", Dict[str, Any]]] = None,
        n_gpu_layers: Optional[int] = None,
        split_mode: Optional[int] = None,
        main_gpu: Optional[int] = None,
        tensor_split: Optional[List[float]] = None,
        vocab_only: Optional[bool] = None,
        use_mmap: Optional[bool] = None,
        use_mlock: Optional[bool] = None,
        kv_overrides: Optional[Dict[str, Union[bool, int, float, str]]] = None,
        n_ctx: Optional[int] = None,
        n_batch: Optional[int] = None,
        n_ubatch: Optional[int] = None,
        n_seq_max: Optional[int] = None,
        n_threads: Optional[int] = None,
        n_threads_batch: Optional[int] = None,
        rope_scaling_type: Optional[int] = None,
        pooling_type: Optional[int] = None,
        attention_type: Optional[int] = None,
        rope_freq_base: Optional[float] = None,
        rope_freq_scale: Optional[float] = None,
        yarn_ext_factor: Optional[float] = None,
        yarn_attn_factor: Optional[float] = None,
        yarn_beta_fast: Optional[float] = None,
        yarn_beta_slow: Optional[float] = None,
        yarn_orig_ctx: Optional[int] = None,
        offload_kqv: Optional[bool] = None,
        flash_attn: Optional[bool] = None,
        op_offload: Optional[bool] = None,
        swa_full: Optional[bool] = None,
        no_perf: Optional[bool] = None,
        type_k: Optional[int] = None,
        type_v: Optional[int] = None,
        kv_unified: bool = True,
        max_seq_len: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        draft_model: Optional[str] = None,
        draft_model_num_pred_tokens: int = 16,
        draft_model_max_ngram_size: int = 2,
        draft_model_top_k: int = 1,
        draft_model_p_min: float = 0.0,
        draft_model_max_batch_size: Optional[int] = None,
        draft_model_threads: Optional[int] = None,
        draft_model_threads_batch: Optional[int] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        store_logits: bool = True,
    ) -> None:
        self._runtime = _load_runtime()
        runtime_loras = self._runtime_loras(loras)
        self._model = self._runtime.Model(
            model_path=model_path,
            model_alias=model_alias,
            chat_template=chat_template,
            loras=runtime_loras,
            n_gpu_layers=n_gpu_layers,
            split_mode=split_mode,
            main_gpu=main_gpu,
            tensor_split=tensor_split,
            vocab_only=vocab_only,
            use_mmap=use_mmap,
            use_mlock=use_mlock,
            kv_overrides=kv_overrides,
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_ubatch=n_ubatch,
            n_seq_max=n_seq_max,
            n_threads=n_threads,
            n_threads_batch=n_threads_batch,
            rope_scaling_type=rope_scaling_type,
            pooling_type=pooling_type,
            attention_type=attention_type,
            rope_freq_base=rope_freq_base,
            rope_freq_scale=rope_freq_scale,
            yarn_ext_factor=yarn_ext_factor,
            yarn_attn_factor=yarn_attn_factor,
            yarn_beta_fast=yarn_beta_fast,
            yarn_beta_slow=yarn_beta_slow,
            yarn_orig_ctx=yarn_orig_ctx,
            offload_kqv=offload_kqv,
            flash_attn=flash_attn,
            op_offload=op_offload,
            swa_full=swa_full,
            no_perf=no_perf,
            type_k=type_k,
            type_v=type_v,
            kv_unified=kv_unified,
            max_seq_len=max_seq_len,
            max_output_tokens=max_output_tokens,
            draft_model=draft_model,
            draft_model_num_pred_tokens=draft_model_num_pred_tokens,
            draft_model_max_ngram_size=draft_model_max_ngram_size,
            draft_model_top_k=draft_model_top_k,
            draft_model_p_min=draft_model_p_min,
            draft_model_max_batch_size=draft_model_max_batch_size,
            draft_model_threads=draft_model_threads,
            draft_model_threads_batch=draft_model_threads_batch,
            response_schema=response_schema,
            store_logits=store_logits,
        )
        self._configure_mtmd(mtmd)
        self._sequence_cache = self._build_sequence_cache(disk_cache)
        scheduler = self._runtime.CompletionScheduler(
            self._model,
            sequence_cache=self._sequence_cache,
        )
        self._service = self._runtime.CompletionService(scheduler)
        self.completions = _CompletionsResource(self)
        self.chat_completions = _ChatCompletionsResource(self)
        self.chat = _ChatResource(self.chat_completions)
        self.responses = _ResponsesResource(self)
        self._closed = False

    @classmethod
    def from_pretrained(
        cls,
        *,
        repo_id: str,
        filename: str,
        additional_files: Optional[List[str]] = None,
        local_dir: Optional[str] = None,
        local_dir_use_symlinks: Union[bool, Literal["auto"]] = "auto",
        cache_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> "Model":
        runtime = _load_runtime()
        options = runtime.ConfigFile.FromPretrainedOptions(
            repo_id=repo_id,
            filename=filename,
            additional_files=additional_files,
            local_dir=local_dir,
            local_dir_use_symlinks=local_dir_use_symlinks,
            cache_dir=cache_dir,
        )
        return cls(model_path=options.resolve_model_path(), **kwargs)

    @property
    def model_path(self) -> str:
        return self._model.model_path

    @property
    def model_alias(self) -> Optional[str]:
        return self._model.model_alias

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._service.close()

    def __enter__(self) -> "Model":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def create_completion(self, **kwargs: Any) -> CompletionResult:
        body = self._runtime.CreateCompletionRequest.model_validate(kwargs)
        prompts = body.normalized_prompt()
        if len(prompts) > 1:
            submissions = [
                self._service.submit(body.model_copy(update={"prompt": prompt}))
                for prompt in prompts
            ]
            try:
                completions = [
                    self._service.formatter.collect_completion(stream)
                    for stream, _ in submissions
                ]
            finally:
                self._close_streams(submissions)
            return _dump_json(
                self._service.formatter.aggregate_completion_results(completions)
            )
        stream, cancel = self._service.submit(
            body.model_copy(update={"prompt": prompts[0]})
        )
        if body.stream:
            return self._completion_stream(stream, cancel)
        return self._collect_completion(stream, cancel)

    def create_chat_completion(self, **kwargs: Any) -> ChatCompletionResult:
        body = self._runtime.CreateChatCompletionRequest.model_validate(kwargs)
        formatter = self._service.formatter
        parts = formatter.completion_request_from_chat_request(body)
        template_functions = (
            [function.to_template_function() for function in body.functions]
            if body.functions is not None
            else None
        )
        template_tools = (
            [tool.to_template_tool() for tool in body.tools]
            if body.tools is not None
            else None
        )
        request = self._service.request_from_prepared(
            payload=parts.payload,
            prompt_text=parts.prompt_text,
            prompt_plan=parts.prompt_plan,
            grammar_text=parts.grammar_text,
            chat_tool_name=parts.tool_name,
        )
        stream, cancel = self._service.submit_request(request)
        if body.stream:
            return self._chat_stream(
                stream,
                cancel,
                parts,
                template_functions=template_functions,
                template_tools=template_tools,
            )
        completion = self._collect_completion_model(stream, cancel)
        return _dump_json(
            formatter.convert_completion_response_to_chat(
                completion,
                parts.tool_name,
                functions=template_functions,
                tools=template_tools,
                generation_prompt=parts.generation_prompt,
            )
        )

    def create_response(self, **kwargs: Any) -> ResponseResult:
        body = self._runtime.CreateResponseRequest.model_validate(kwargs)
        formatter = self._service.formatter
        chat_parts = formatter.response_request_to_chat_parts(body)
        parts = formatter.completion_request_from_response_chat_parts(chat_parts)
        response_tools = chat_parts.tools
        request = self._service.request_from_prepared(
            payload=parts.payload,
            prompt_text=parts.prompt_text,
            prompt_plan=parts.prompt_plan,
            grammar_text=parts.grammar_text,
            chat_tool_name=parts.tool_name,
        )
        stream, cancel = self._service.submit_request(request)
        if body.stream:
            return self._response_stream(
                stream, cancel, body, request, parts, response_tools
            )
        completion = self._collect_completion_model(stream, cancel)
        return _dump_json(
            formatter.convert_completion_response_to_response(
                completion,
                body,
                parts.tool_name,
                tools=response_tools,
                generation_prompt=parts.generation_prompt,
            )
        )

    def render_prometheus_metrics(self) -> str:
        return self._service.render_prometheus_metrics()

    def _runtime_loras(
        self,
        loras: Optional[Sequence[Union["Model.LoraAdapter", Dict[str, Any]]]],
    ) -> Optional[List[Any]]:
        if loras is None:
            return None
        result: List[Any] = []
        for lora in loras:
            if isinstance(lora, dict):
                lora = Model.LoraAdapter(**lora)
            result.append(
                self._runtime.Model.LoraAdapter(path=lora.path, scale=lora.scale)
            )
        return result

    def _configure_mtmd(
        self,
        mtmd: Optional[Union["Model.MTMDConfig", Dict[str, Any]]],
    ) -> None:
        if mtmd is None:
            return
        if isinstance(mtmd, dict):
            mtmd = Model.MTMDConfig(**mtmd)
        if self._model.chat_formatter is None:
            raise RuntimeError("MTMD requires a GGUF chat template")
        embedding_cache = None
        if mtmd.embedding_cache_path is not None:
            embedding_cache = self._runtime.MTMDEmbeddingCache(
                path=mtmd.embedding_cache_path,
                max_bytes=mtmd.embedding_cache_max_bytes,
                model_fingerprint=self._runtime.MTMDEmbeddingCache.fingerprint_file(
                    self._model.model_path,
                ),
                mmproj_fingerprint=self._runtime.MTMDEmbeddingCache.fingerprint_file(
                    mtmd.mmproj_path,
                ),
            )
        self._model.mtmd_processor = self._runtime.MTMDProcessor(
            model_path=self._model.model_path,
            llama_model=self._model.llama_model,
            chat_formatter=self._model.chat_formatter,
            tokenize=self._model.tokenize,
            n_embd_inp=self._model.n_embd_inp,
            n_batch=self._model.n_batch,
            n_ubatch=self._model.n_ubatch,
            n_threads_batch=self._model.n_threads_batch,
            mmproj_path=mtmd.mmproj_path,
            embedding_cache=embedding_cache,
            allowed_media_domains=mtmd.allowed_media_domains,
            allowed_local_media_path=mtmd.allowed_local_media_path,
            image_max_bytes=mtmd.image_max_bytes,
            audio_max_bytes=mtmd.audio_max_bytes,
            image_timeout_seconds=mtmd.image_timeout_seconds,
        )

    def _build_sequence_cache(
        self,
        disk_cache: Optional[Union["Model.DiskCacheConfig", Dict[str, Any]]],
    ) -> Optional[Any]:
        if disk_cache is None:
            return None
        if isinstance(disk_cache, dict):
            disk_cache = Model.DiskCacheConfig(**disk_cache)
        return self._runtime.SequenceDiskCache(
            path=disk_cache.path,
            max_bytes=disk_cache.max_bytes,
            min_tokens=disk_cache.min_tokens,
            compatibility_key=self._runtime.SequenceDiskCache.compatibility_key_for_model(
                self._model,
            ),
        )

    def _collect_completion_model(self, stream: Any, cancel: Callable[[], None]) -> Any:
        try:
            return self._service.formatter.collect_completion(stream)
        except BaseException:
            cancel()
            raise
        finally:
            stream.close()

    def _collect_completion(
        self, stream: Any, cancel: Callable[[], None]
    ) -> Dict[str, Any]:
        return _dump_json(self._collect_completion_model(stream, cancel))

    @staticmethod
    def _close_streams(submissions: Sequence[Tuple[Any, Callable[[], None]]]) -> None:
        for stream, _ in submissions:
            stream.close()

    @staticmethod
    def _completion_stream(
        stream: Any,
        cancel: Callable[[], None],
    ) -> Generator[Dict[str, Any], None, None]:
        try:
            for chunk in stream:
                yield _dump_json(chunk)
        except BaseException:
            cancel()
            raise
        finally:
            stream.close()

    def _chat_stream(
        self,
        stream: Any,
        cancel: Callable[[], None],
        parts: Any,
        *,
        template_functions: Optional[List[Any]],
        template_tools: Optional[List[Any]],
    ) -> Generator[Dict[str, Any], None, None]:
        formatter = self._service.formatter
        started_indices: set[int] = set()
        parsed_states: Dict[int, Dict[str, Any]] = {}
        try:
            for completion_chunk in stream:
                for payload in formatter.convert_completion_chunk_to_chat_chunks(
                    completion_chunk,
                    started_indices,
                    parts.tool_name,
                    functions=template_functions,
                    tools=template_tools,
                    parsed_states=parsed_states,
                    generation_prompt=parts.generation_prompt,
                ):
                    yield _dump_json(payload)
        except BaseException:
            cancel()
            raise
        finally:
            stream.close()

    def _response_stream(
        self,
        stream: Any,
        cancel: Callable[[], None],
        body: Any,
        request: Any,
        parts: Any,
        response_tools: Optional[List[Any]],
    ) -> Generator[Dict[str, Any], None, None]:
        formatter = self._service.formatter
        started_indices: set[int] = set()
        parsed_states: Dict[int, Any] = {}
        stream_state = self._runtime.OpenAIFormatter.ResponsesStream(
            body=body,
            response_id="resp_" + request.id,
            created_at=float(request.created),
            model=self._service.scheduler.model.model_path,
        )
        try:
            for payload in formatter.start_response_stream(stream_state):
                yield _dump_json(payload)
            while True:
                done, completion_chunk, completion = formatter.next_stream_output(
                    stream
                )
                if done:
                    for payload in formatter.response_stream_terminal_events(
                        stream_state,
                        completion,
                    ):
                        yield _dump_json(payload)
                    break
                assert completion_chunk is not None
                chat_chunks = formatter.convert_completion_chunk_to_chat_chunks(
                    completion_chunk,
                    started_indices,
                    parts.tool_name,
                    tools=response_tools,
                    parsed_states=parsed_states,
                    generation_prompt=parts.generation_prompt,
                )
                for chat_chunk in chat_chunks:
                    for payload in formatter.convert_chat_chunk_to_response_events(
                        chat_chunk,
                        stream_state,
                    ):
                        yield _dump_json(payload)
        except BaseException:
            cancel()
            raise
        finally:
            stream.close()
