import os
import sys
import unittest
from model_infer import test_model_inference
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class TestChatglm2Infer(unittest.TestCase):

    def test_chatglm2_infer(self):
        from dancingmodel.models.chatglm2.model import ChatGlm2TpPartModel
        test_model_inference(world_size=1, 
                             model_dir="/nvme/baishihao/chatglm2-6b/", 
                             model_class=ChatGlm2TpPartModel, 
                             batch_size=20, 
                             input_len=1024, 
                             output_len=1024)
        return


if __name__ == '__main__':
    unittest.main()