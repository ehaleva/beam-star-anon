# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.openai.engine.serving import OpenAIServing
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from beam_star.beam_search import beam_search
from beam_star.protocol import to_beam_search_params
from beam_star.stats_utils import GlobalRunningStats


def patch_beam_search():
    original_init = OpenAIServing.__init__  
  
    def patched_init(self, *args, **kwargs):  
        # Extract and store cum_logprob_threshold if provided 
        
        self.req_stats = [GlobalRunningStats(), GlobalRunningStats(), GlobalRunningStats(), GlobalRunningStats()]
        self.thr = [-20.0, -8.0, -8.0, -8.0]  # default threshold values for the two phases        
        original_init(self, *args, **kwargs)  

    OpenAIServing.__init__ = patched_init  

    OpenAIServing.beam_search = beam_search

    ChatCompletionRequest.to_beam_search_params = to_beam_search_params

