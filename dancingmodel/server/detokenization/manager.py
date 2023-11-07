import uvloop
import asyncio
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import zmq
import zmq.asyncio
from ..io_struct import BatchTokenIdOut, ReqDetokenizationState, BatchStrOut, AbortReq, BatchAbortReq
from typing import Union
from .decode import decode_token
from ..tokenizer import get_tokenizer
import traceback

from dancingmodel.utils.infer_utils import calculate_time, mark_start, mark_end

class DeTokenizationManager:
    
    def __init__(self, model_weightdir, tokenizor_mode, detokenization_port, httpserver_port, trust_remote_code,
                 dummy=False):
        context = zmq.asyncio.Context(2)
        self.recv_from_router = context.socket(zmq.PULL)
        self.recv_from_router.bind(f"tcp://127.0.0.1:{detokenization_port}")
        
        self.send_to_httpserver = context.socket(zmq.PUSH)
        self.send_to_httpserver.connect(f"tcp://127.0.0.1:{httpserver_port}")
        
        try: 
            self.tokenizer = get_tokenizer(model_weightdir, tokenizor_mode, trust_remote_code=trust_remote_code) 
        except:
            if dummy:
                self.tokenizer = get_tokenizer("huggyllama/llama-7b", tokenizor_mode) 

        self.req_id_to_out = {}
        
    async def handle_loop(self):
        while True:
            try:
                recv_obj:Union(BatchTokenIdOut, ReqDetokenizationState, AbortReq, BatchAbortReq) = await self.recv_from_router.recv_pyobj() 
                assert isinstance(recv_obj, (BatchTokenIdOut, ReqDetokenizationState, AbortReq, BatchAbortReq)), f"type is not right {type(recv_obj)}"
                if isinstance(recv_obj, ReqDetokenizationState):
                    self.req_id_to_out[recv_obj.request_id] = recv_obj
                
                if isinstance(recv_obj, AbortReq):
                    delete_req_id = recv_obj.req_id
                    if delete_req_id in self.req_id_to_out:
                        del self.req_id_to_out[delete_req_id]
                
                if isinstance(recv_obj, BatchAbortReq):
                    for delete_req_id in recv_obj.reqs:
                        if delete_req_id in self.req_id_to_out:
                            del self.req_id_to_out[delete_req_id]
                    self.send_to_httpserver.send_pyobj(recv_obj)
                        
                if isinstance(recv_obj, BatchTokenIdOut):
                    new_batch_str_out = BatchStrOut()
                    for req_id, new_token_id, new_gen_metadata, finished, abort in recv_obj.reqs_infs:
                        if req_id not in self.req_id_to_out:
                            continue
                        req_out:ReqDetokenizationState = self.req_id_to_out[req_id]
                        req_out.output_ids.append(new_token_id)
                        req_out.gen_metadata.update(new_gen_metadata)
                        out_text = decode_token(self.tokenizer, req_out, new_token_id, skip_special_tokens=True)
                        if out_text.endswith(u'\ufffd'):
                            new_text = ''
                        else:
                            new_text = out_text[len(req_out.output_str):]
                            req_out.output_str = out_text
                        new_batch_str_out.reqs_infs.append((req_id, new_text, new_gen_metadata, True if abort else finished, abort))
                        if finished or abort:
                            try:
                                del self.req_id_to_out[req_id]
                            except:
                                pass
                    self.send_to_httpserver.send_pyobj(new_batch_str_out)
            except Exception as e:
                print(f"detoken process has exception {str(e)}")
                traceback.print_exc()
                pass


def start_detokenization_process(args, detokenization_port, httpserver_port, pipe_writer, trust_remote_code):
    try:
        router = DeTokenizationManager(args.model_dir, args.tokenizer_mode,
                                       detokenization_port=detokenization_port, httpserver_port=httpserver_port,
                                       trust_remote_code=trust_remote_code, dummy=args.dummy)
    except Exception as e:
        pipe_writer.send(str(e))
        raise
    pipe_writer.send('init ok')
    loop = asyncio.get_event_loop()
    loop.run_until_complete(router.handle_loop())
    return
