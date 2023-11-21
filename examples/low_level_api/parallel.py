"""
parallel.cpp's python version
https://github.com/ggerganov/llama.cpp/blob/master/examples/parallel/parallel.cpp

notice:
not support grammar
not support terminal args
not support prompt file
"""

import ctypes
import dataclasses
import random
import time

from llama_cpp import llama_cpp, Llama, LlamaGrammar

from typing import Dict, Optional, Tuple

from llama_cpp.llama import _LlamaContext, _LlamaBatch
from llama_cpp.llama_grammar import std, parse_state

from dataclasses import dataclass

k_system = """Transcript of a never ending dialog, where the User interacts with an Assistant.
The Assistant is helpful, kind, honest, good at writing, and never fails to answer the User's requests immediately and with precision.

User: Recommend a nice restaurant in the area.
Assistant: I recommend the restaurant "The Golden Duck". It is a 5 star restaurant with a great view of the city. The food is delicious and the service is excellent. The prices are reasonable and the portions are generous. The restaurant is located at 123 Main Street, New York, NY 10001. The phone number is (212) 555-1234. The hours are Monday through Friday from 11:00 am to 10:00 pm. The restaurant is closed on Saturdays and Sundays.
User: Who is Richard Feynman?
Assistant: Richard Feynman was an American physicist who is best known for his work in quantum mechanics and particle physics. He was awarded the Nobel Prize in Physics in 1965 for his contributions to the development of quantum electrodynamics. He was a popular lecturer and author, and he wrote several books, including "Surely You're Joking, Mr. Feynman!" and "What Do You Care What Other People Think?".
User:
"""

k_prompts = [
    "What is the meaning of life?",
    "Tell me an interesting fact about llamas.",
    "What is the best way to cook a steak?",
    "Are you familiar with the Special Theory of Relativity and can you explain it to me?",
    "Recommend some interesting books to read.",
    "What is the best way to learn a new language?",
    "How to get a job at Google?",
    "If you could have any superpower, what would it be?",
    "I want to learn how to play the piano.",
]


@dataclass
class LlamaSamplingParams(object):
    n_prev: int = 10  # number of previous tokens to remember
    n_probs: int = 0  # if greater than 0, output the probabilities of top n_probs tokens.
    top_k: int = 40  # <= 0 to use vocab size
    top_p: float = 0.95  # 1.0 = disabled
    min_p: float = 0.05  # 0.0 = disabled
    tfs_z: float = 1.00  # 1.0 = disabled
    typical_p: float = 1.00  # 1.0 = disabled
    temp: float = 0.80  # 1.0 = disabled
    penalty_last_n: int = 64  # last n tokens to penalize (0 = disable penalty, -1 = context size)
    penalty_repeat: float = 1.10  # 1.0 = disabled
    penalty_freq: float = 0.00  # 0.0 = disabled
    penalty_present: float = 0.00  # 0.0 = disabled
    mirostat: int = 0  # 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
    mirostat_tau: float = 5.00  # target entropy
    mirostat_eta: float = 0.10  # learning rate
    penalize_nl: bool = True  # consider newlines as a repeatable token

    grammar: Optional[str] = None  # optional BNF-like grammar to constrain sampling

    # Classifier-Free Guidance
    # https://arxiv.org/abs/2306.17806
    cfg_negative_prompt: Optional[str] = None  # string to help guidance
    cfg_scale: float = 1.0  # how strong is guidance

    logit_bias: Optional[Dict[int, float]] = None  # logit bias for specific tokens


@dataclass
class LlamaSamplingContext(object):
    params: LlamaSamplingParams  # parameters that will be used for sampling
    parsed_grammar: Optional[parse_state]  # internal
    prev: std.vector[llama_cpp.llama_token]  # TODO: replace with ring-buffer
    cur: std.vector[llama_cpp.llama_token_data]

    grammar: Optional[LlamaGrammar]  # llama_grammar
    mirostat_mu: Optional[float] = None  # mirostat sampler state


def llama_sampling_reset(ctx: LlamaSamplingContext):
    if ctx.grammar is not None:
        ctx.grammar.reset()

    ctx.prev.resize(ctx.params.n_prev, lambda: llama_cpp.llama_token(0))
    ctx.cur.clear()


def llama_sampling_sample(ctx_sampling: LlamaSamplingContext,
                          ctx_main: _LlamaContext, ctx_cfg, idx: int):
    """
    # Based on https://github.com/ggerganov/llama.cpp/blob/master/examples/common.cpp
    llama_sampling_sample
    """
    params = ctx_sampling.params

    n_vocab = ctx_main.model.n_vocab()

    temp = params.temp
    top_k = params.top_k if params.top_k > 0 else n_vocab
    top_p = params.top_p
    min_p = params.min_p
    tfs_z = params.tfs_z
    typical_p = params.typical_p
    penalty_last_n = params.penalty_last_n if params.penalty_last_n >= 0 else params.n_prev
    penalty_repeat = params.penalty_repeat
    penalty_freq = params.penalty_freq
    penalty_present = params.penalty_present
    mirostat = params.mirostat
    mirostat_tau = params.mirostat_tau
    mirostat_eta = params.mirostat_eta
    penalize_nl = params.penalize_nl

    prev = ctx_sampling.prev
    cur = ctx_sampling.cur

    # id = 0

    logits = llama_cpp.llama_get_logits_ith(ctx_main.ctx, idx)

    # baii fix logit_bias is None
    if params.logit_bias:
        for key, value in params.logit_bias.items():
            logits[key] += value

    # 性能优化换用 numpy 数组实现
    cur.clear()
    # cur = None
    #
    cur = (llama_cpp.llama_token_data * n_vocab)()

    for token_id in range(n_vocab):
        cur[token_id].id = token_id
        cur[token_id].logit = logits[token_id]
        cur[token_id].p = 0.0

    cur_p = ctypes.byref(llama_cpp.llama_token_data_array(cur, n_vocab, False))

    if ctx_cfg:
        llama_cpp.llama_sample_classifier_free_guidance(ctx_main.ctx, cur_p, ctx_cfg, params.cfg_scale)

    if prev:
        # 等价于llama.cpp的
        # last_n_tokens_data = [llama_cpp.llama_token(0)] * max(
        #     0, self.last_n_tokens_size - self.n_tokens
        # ) + self._input_ids[-self.last_n_tokens_size:].tolist()
        last_n_tokens_data = (llama_cpp.llama_token * penalty_last_n)(
            *[i for i in prev.iterator(prev, 0)][-penalty_last_n:])

        nl_logit = logits[ctx_main.model.token_nl()]
        llama_cpp.llama_sample_repetition_penalties(
            ctx_main.ctx,
            cur_p,
            last_n_tokens_data,
            penalty_last_n,
            penalty_repeat,
            penalty_freq, penalty_present)

        if not penalize_nl:
            for item in cur_p['data']:
                if item['id'] == ctx_main.model.token_nl():
                    item['logit'] = nl_logit
                    break

    if ctx_sampling.grammar:
        llama_cpp.llama_sample_grammar(ctx_main.ctx, cur_p, ctx_sampling.grammar.grammar)

    if temp < 0.0:
        # greedy sampling, with probs
        llama_cpp.llama_sample_softmax(ctx_main.ctx, cur_p)
        id = cur_p['data'][0]['id']
    elif temp == 0.0:
        # greedy sampling, no probs
        id = llama_cpp.llama_sample_token_greedy(ctx_main.ctx, cur_p)
    else:
        if mirostat == 1:
            mirostat_m = 100
            llama_cpp.llama_sample_temp(ctx_main.ctx, cur_p, temp)
            id = llama_cpp.llama_sample_token_mirostat(ctx_main.ctx, cur_p, mirostat_tau, mirostat_eta, mirostat_m,
                                                       ctx_sampling.mirostat_mu)
        elif mirostat == 2:
            llama_cpp.llama_sample_temp(ctx_main.ctx, cur_p, temp)
            id = llama_cpp.llama_sample_token_mirostat_v2(ctx_main.ctx, cur_p, mirostat_tau, mirostat_eta,
                                                          ctx_sampling.mirostat_mu)
        else:
            # temperature sampling
            min_keep = max(1, params.n_probs)

            llama_cpp.llama_sample_top_k(ctx_main.ctx, cur_p, top_k, min_keep)
            llama_cpp.llama_sample_tail_free(ctx_main.ctx, cur_p, tfs_z, min_keep)
            llama_cpp.llama_sample_typical(ctx_main.ctx, cur_p, typical_p, min_keep)
            llama_cpp.llama_sample_top_p(ctx_main.ctx, cur_p, top_p, min_keep)
            llama_cpp.llama_sample_min_p(ctx_main.ctx, cur_p, min_p, min_keep)
            llama_cpp.llama_sample_temp(ctx_main.ctx, cur_p, temp)

            id = llama_cpp.llama_sample_token(ctx_main.ctx, cur_p)

            # Log top candidates
            # n_top = 10
            # print("Top {} candidates:".format(n_top))
            # for i in range(min(n_top, len(cur_p['data']))):
            #     print(" - {}: '{}' ({:.3f})".format(cur_p['data'][i]['id'],
            #                                           llama_token_to_piece(ctx_main, cur_p['data'][i]['id']),
            #                                           cur_p['data'][i]['p']))

            # size = 32
            # print("Sampled token: {}: '{token_str}'".format(
            #     id,
            #     token_str=ctx_main.model.detokenize([id]).decode('utf-8',
            #                                                      'ignore')))

    return id


def llama_sampling_accept(
        ctx_sampling: LlamaSamplingContext,
        ctx_main: _LlamaContext,
        token_id: int, apply_grammar: bool):
    ctx_sampling.prev.pop(0)
    ctx_sampling.prev.append(token_id)

    if ctx_sampling.grammar and apply_grammar:
        llama_cpp.llama_grammar_accept_token(ctx_main.ctx, ctx_sampling.grammar.grammar, id)


def _llama_batch_add(batch: _LlamaBatch, token, pos, seq_ids, logits):
    real_batch = batch.batch
    real_batch.token[real_batch.n_tokens] = llama_cpp.llama_token(token)
    real_batch.pos[real_batch.n_tokens] = pos
    real_batch.n_seq_id[real_batch.n_tokens] = len(seq_ids)

    for i in range(len(seq_ids)):
        real_batch.seq_id[real_batch.n_tokens][i] = seq_ids[i]

    real_batch.logits[real_batch.n_tokens] = logits

    real_batch.n_tokens += 1
    batch.n_tokens = batch.batch.n_tokens


def ggml_time_us() -> float:
    return time.time() * 1000000


def llama_sampling_free(ctx: LlamaSamplingContext):
    if ctx.grammar is not None:
        del ctx.grammar
        # llama_cpp.llama_grammar_free(ctx.grammar)
    del ctx


def llama_batch_clear(batch: _LlamaBatch):
    batch.n_tokens = 0
    batch.batch.n_tokens = 0


def _get_batch_view(batch: _LlamaBatch, n_tokens: int, offset: int) -> llama_cpp.llama_batch:
    """
    like cpp code:
            llama_batch batch_view = {
                n_tokens,
                batch.token    + i,
                nullptr,
                batch.pos      + i,
                batch.n_seq_id + i,
                batch.seq_id   + i,
                batch.logits   + i,
                0, 0, 0, // unused
            };
    :param n_tokens:
    :param offset:
    :return:
    """

    def _move_pointer_offset(ptr, c_types, offset: int):
        """
        Move the pointer (pointer counts)
        :param ptr: 要移动的指针
        :param c_types: 指针指向内存的类型
        :param offset: 移动的偏移量
        :return:
        """
        return ctypes.cast(
            ctypes.addressof(ptr.contents) + offset * ctypes.sizeof(c_types),
            ctypes.POINTER(c_types)
        )

    batch_view = llama_cpp.llama_batch(
        n_tokens=n_tokens,
        token=_move_pointer_offset(batch.batch.token, llama_cpp.llama_token, offset),
        embd=None,
        pos=_move_pointer_offset(batch.batch.pos, llama_cpp.llama_pos, offset),
        n_seq_id=_move_pointer_offset(batch.batch.n_seq_id, ctypes.c_int32, offset),
        seq_id=_move_pointer_offset(batch.batch.seq_id, ctypes.POINTER(llama_cpp.llama_seq_id), offset),
        logits=_move_pointer_offset(batch.batch.logits, ctypes.c_int8, offset),
        all_pos_0=0,
        all_pos_1=0,
        all_seq_id=0,
    )
    return batch_view


def llama_sampling_init(params: LlamaSamplingParams):
    result = LlamaSamplingContext(
        params=LlamaSamplingParams(),
        grammar=params.grammar,
        parsed_grammar=parse_state(),
        prev=std.vector(),
        cur=std.vector(),
    )

    result.params = params
    result.grammar = None

    # if there is a grammar, parse it
    if params.grammar:
        # result.parsed_grammar = LlamaGrammar.from_string(params.grammar)
        # if not result.parsed_grammar.grammar.rules:
        #     print(f"{__func__}: failed to parse grammar")
        #     return None
        #
        # grammar_rules = result.parsed_grammar.grammar.c_rules()
        #
        # result.grammar = llama_cpp.llama_grammar_init(
        #     grammar_rules.data(),
        #     grammar_rules.size(),
        #     result.parsed_grammar.grammar.symbol_ids["root"]
        # )
        # result.parsed_grammar = LlamaGrammar.from_string(params.grammar)
        result.grammar = LlamaGrammar.from_string(params.grammar)
    result.prev.resize(params.n_prev, lambda: 0)
    return result


@dataclass
class Client(object):
    # re mame id to unique_id
    unique_id: int
    ctx_sampling: LlamaSamplingContext

    seq_id: int = -1
    sampled: any = None
    t_start_prompt: int = 0
    t_start_gen: int = 0
    n_prompt: int = 0
    n_decoded: int = 0
    i_batch: int = -1
    input: str = ""
    prompt: str = ""
    response: str = ""
    decode_err_buffer: bytes = b''

    def __del__(self):
        if self.ctx_sampling:
            llama_sampling_free(self.ctx_sampling)


def process_system_prompt(llama: Llama) -> Tuple[int, float]:
    # tokenize system prompt
    tokens_system = llama.tokenize(
        k_system.encode(), True
    )
    n_tokens_system = len(tokens_system)

    batch = llama._batch

    t_main_start = ggml_time_us()

    print("Simulating parallel requests from clients:", )
    print(
        f"n_parallel = {n_parallel}, n_sequences = {n_seq}, cont_batching = {cont_batching}, system tokens = {n_tokens_system}")
    print("Evaluating the system prompt ...")

    for i in range(n_tokens_system):
        _llama_batch_add(batch, tokens_system[i], i, [0], False)

    if llama_cpp.llama_decode(llama._ctx.ctx, batch.batch) != 0:
        print("llama_decode() failed")
        exit(1)

    # 将系统 KV 缓存分配给所有并行序列
    for i in range(1, n_parallel):
        llama_cpp.llama_kv_cache_seq_cp(llama._ctx.ctx, 0, i, 0, n_tokens_system)

    return n_tokens_system, t_main_start


# like: llama.cpp common/common.h gpt_params
@dataclasses.dataclass
class GptParams():
    seed: int = 1234
    n_threads: int = ...
    n_ctx: int = 4096 * 4
    n_batch: int = 4096
    n_predict: int = -1
    sparams: LlamaSamplingParams = ...

    model: str = ...
    model_draft: str = ""
    model_alias: str = "unknown"
    prompt: str = ""
    prompt_file: str = ""
    path_prompt_cache: str = ""
    input_prefix: str = ""
    input_suffix: str = ""
    # antiprompt: List[str] = Field(default_factory=list)
    logdir: str = ""
    # lora_adapter: List[Tuple[str, float]] = Field(default_factory=list)
    lora_base: str = ""
    ppl_stride: int = 0
    ppl_output_type: int = 0
    hellaswag: bool = False
    hellaswag_tasks: int = 400
    mul_mat_q: bool = True
    memory_f16: bool = True
    random_prompt: bool = False
    use_color: bool = False
    interactive: bool = False
    prompt_cache_all: bool = False
    prompt_cache_ro: bool = False
    embedding: bool = False
    escape: bool = False
    interactive_first: bool = False
    multiline_input: bool = False
    simple_io: bool = False
    cont_batching: bool = True
    input_prefix_bos: bool = False
    ignore_eos: bool = False
    instruct: bool = False
    logits_all: bool = False
    use_mmap: bool = True
    use_mlock: bool = False
    numa: bool = False
    verbose_prompt: bool = False
    infill: bool = False
    mmproj: str = ""
    image: str = ""
    n_sequences: int = 10


if __name__ == '__main__':

    # initialization parameters
    gpt_params = GptParams(
        n_threads=4,
        model="./ggml-model-q8_0.gguf",
        n_ctx=4096 * 4,
        n_batch=4096,
        # a total of several sentences need to be completed
        n_sequences=1000,
        sparams=LlamaSamplingParams(),
    )
    # some configuration of dynamic batching
    # several in parallel
    n_parallel = 10
    # a total of several sentences need to be completed
    n_seq = gpt_params.n_sequences
    # whether it is a dynamic batch mode
    cont_batching = gpt_params.cont_batching

    llama = Llama(
        model_path=gpt_params.model,
        n_threads=gpt_params.n_threads,
        seed=gpt_params.seed,
        n_ctx=gpt_params.n_ctx,
        n_batch=gpt_params.n_batch,
        # n_predict=gpt_params.n_predict,
        n_gpu_layers=100
    )

    del llama._batch
    # the max batch size is as large as the context to handle cases where we get very long input prompt from multiple
    # users. regardless of the size, the main loop will chunk the batch into a maximum of params.n_batch tokens at a time
    # Dynamic batching uses n_seq_max=1 instead of n_seq_max=n_ctx
    llama._batch = _LlamaBatch(
        n_tokens=llama.context_params.n_ctx,
        embd=0,
        n_seq_max=1,
        verbose=llama.verbose,
    )

    if not gpt_params.prompt:
        print("\033[32mNo new questions so proceed with build-in defaults.\033[0m")
    else:
        # Output each line of the input gpt_params.prompt vector and copy to k_prompts
        index = 0
        print("\033[32mNow printing the external prompt file {}\033[0m".format(gpt_params.prompt_file))

        prompts = gpt_params.prompt.split('\n')
        k_prompts = []

        for prompt in prompts:
            k_prompts.append(prompt)
            index += 1
            print("{:3d} prompt: {}".format(index, prompt))

    # n_ctx = llama_instance.n_ctx()

    clients = [
        Client(
            unique_id=i,
            ctx_sampling=llama_sampling_init(gpt_params.sparams)
        ) for i in range(n_parallel)
    ]

    n_tokens_system, t_main_start = process_system_prompt(llama)

    g_seq_id = 0

    n_total_prompt = 0
    n_total_gen = 0
    n_cache_miss = 0

    print("Processing requests ...")

    ctx: _LlamaContext = llama._ctx
    batch: _LlamaBatch = llama._batch

    while True:
        llama_batch_clear(batch)

        # Decode any currently ongoing sequences
        for client in clients:
            if client.seq_id == -1:
                continue

            client.i_batch = batch.n_tokens

            _llama_batch_add(
                batch,
                client.sampled, n_tokens_system + client.n_prompt + client.n_decoded,
                [client.unique_id],
                True)
            client.n_decoded += 1

        if batch.n_tokens == 0:
            # All sequences have ended - clear the entire KV cache
            for i in range(len(clients)):
                llama_cpp.llama_kv_cache_seq_rm(ctx.ctx, i, n_tokens_system, -1)

            print("clearing the KV cache")

        # Insert new sequences for decoding
        if cont_batching or batch.n_tokens == 0:
            for client in clients:
                if client.seq_id == -1 and g_seq_id < n_seq:
                    client.seq_id = g_seq_id

                    client.t_start_prompt = ggml_time_us()
                    client.t_start_gen = 0

                    client.input = random.choice(k_prompts)
                    client.prompt = client.input + "\nAssistant: "
                    client.response = ""

                    llama_sampling_reset(client.ctx_sampling)

                    # Do not prepend BOS because we have a system prompt!
                    tokens_prompt = llama.tokenize(
                        client.prompt.encode(),
                        False, False
                    )

                    for i, token_prompt in enumerate(tokens_prompt):
                        _llama_batch_add(
                            batch,
                            token_prompt, i + n_tokens_system,
                            [client.unique_id],
                            False)

                    # Extract the logits only for the last token
                    if batch.n_tokens > 0:
                        batch.batch.logits[batch.n_tokens - 1] = True

                    client.n_prompt = len(tokens_prompt)
                    client.n_decoded = 0
                    client.i_batch = batch.n_tokens - 1

                    print(
                        f"\033[31mClient {client.unique_id}, seq {client.seq_id}, n_prompt={client.n_prompt} started decoding ...\033[0m")
                    g_seq_id += 1

        if batch.n_tokens == 0:
            break

        for i in range(0, batch.n_tokens, gpt_params.n_batch):
            n_tokens = min(gpt_params.n_batch, batch.n_tokens - i)
            batch_view = _get_batch_view(batch, n_tokens, i)
            ret = llama_cpp.llama_decode(ctx.ctx, batch_view)

            if ret != 0:
                if gpt_params.n_batch == 1 or ret < 0:
                    print(f"failed to decode the batch, n_batch = {gpt_params.n_batch}, ret = {ret}")
                    exit(1)

                print(f"failed to decode the batch, retrying with n_batch = {gpt_params.n_batch // 2}")

                n_cache_miss += 1

                # Retry with half the batch size to try to find a free slot in the KV cache
                gpt_params.n_batch //= 2
                i -= gpt_params.n_batch
                continue

            for client in clients:
                if client.i_batch < i or client.i_batch >= i + n_tokens:
                    continue
                token_id = llama_sampling_sample(client.ctx_sampling, ctx, None, client.i_batch - i)

                llama_sampling_accept(client.ctx_sampling, ctx, token_id, True)

                if client.n_decoded == 1:
                    # Start measuring generation time after the first token to make sure all concurrent clients
                    # have their prompt already processed
                    client.t_start_gen = ggml_time_us()

                size = 32
                token_str = llama.detokenize([token_id])
                # simple decode support zh-cn
                try:
                    client.response += (client.decode_err_buffer + token_str).decode('utf8')
                    client.decode_err_buffer = b''
                except UnicodeDecodeError:
                    client.decode_err_buffer += token_str
                    # print(f'{id=} {token_str} 解码失败')
                # client.response += token_str.decode('utf8', 'replace')
                # print(f"\033[31mClient {client.id}, seq {client.seq_id}, response {client.response}, \033[0m")
                client.sampled = token_id  # print(sw.pretty_print())

                if client.n_decoded > 2 and (
                        token_id == llama.token_eos() or
                        (gpt_params.n_predict > 0 and client.n_decoded >= 10) or
                        "User:" in client.response or
                        '\n' in client.response
                ):
                    # Basic reverse prompt
                    pos = client.response.find("User:")
                    if pos != -1:
                        client.response = client.response[:pos]

                    # Delete only the generated part of the sequence, i.e., keep the system prompt in the cache
                    llama_cpp.llama_kv_cache_seq_rm(ctx.ctx, client.unique_id, n_tokens_system, -1)
                    t_main_end = ggml_time_us()

                    print(
                        f"\033[31mClient {client.unique_id}, seq {client.seq_id}/{n_seq}, prompt {client.n_prompt} t, "
                        f"response {client.n_decoded} t, time {(t_main_end - client.t_start_prompt) / 1e6} s, "
                        f"speed {(client.n_prompt + client.n_decoded) / (t_main_end - client.t_start_prompt) * 1e6} t/s, "
                        f"cache miss {n_cache_miss}\033[0m \nInput:    {client.input}\n"
                        f"\033[35mResponse: {client.response}\033[0m")

                    n_total_prompt += client.n_prompt
                    n_total_gen += client.n_decoded

                    client.seq_id = -1

                client.i_batch = -1

    t_main_end = ggml_time_us()
    print("n_parallel = %d, n_sequences = %d, cont_batching = %d, system tokens = %d" % (
        n_parallel, n_seq, cont_batching, n_tokens_system))
    # notice: no support get prompt file
    prompt_file = "used built-in defaults"
    print(f"External prompt file: \033[32m{prompt_file}\033[0m")
    print(f"Model and path used:  \033[32m{gpt_params.model}\033[0m\n")

    print("Total prompt tokens: %6d, speed: %5.2f t/s" % (
        n_total_prompt, (n_total_prompt) / (t_main_end - t_main_start) * 1e6))

    print("Total gen tokens:    %6d, speed: %5.2f t/s" % (
        n_total_gen, (n_total_gen) / (t_main_end - t_main_start) * 1e6))
    print("Total speed (AVG):   %6s  speed: %5.2f t/s" % (
        "", (n_total_prompt + n_total_gen) / (t_main_end - t_main_start) * 1e6))
    print(f"Cache misses:        {n_cache_miss}\n")
    ctx.print_timings()
