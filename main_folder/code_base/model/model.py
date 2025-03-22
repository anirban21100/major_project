import tensorflow as tf
from tensorflow import keras

from code_base.pipeline.preblock.pre_block import PreBlock

from code_base.pipeline.block_1.block_1 import Block1
from code_base.pipeline.block_1.eca import ECALayer1
from code_base.pipeline.block_1.concat import Concat1
from code_base.pipeline.block_1.dcl import DCL1

from code_base.pipeline.block_2.block_2 import Block2
from code_base.pipeline.block_2.eca import ECALayer2
from code_base.pipeline.block_2.concat import Concat2
from code_base.pipeline.block_2.dcl import DCL2

from code_base.pipeline.block_3.block_3 import Block3
from code_base.pipeline.block_3.eca import ECALayer3
from code_base.pipeline.block_3.concat import Concat3
from code_base.pipeline.block_3.dcl import DCL3

from code_base.pipeline.block_4.block_4 import Block4
from code_base.pipeline.block_4.eca import ECALayer4
from code_base.pipeline.block_4.concat import Concat4
from code_base.pipeline.block_4.dcl import DCL4

from code_base.pipeline.postblock.post_block import PostBlock
from code_base.utils.ArcFace import ArcMarginProduct


class HARModel(tf.keras.Model):
    def __init__(self, num_classes=15, *args, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes

    def build(self):
        self.PreBlock = PreBlock()

        self.block1 = Block1()
        self.block1_eca = ECALayer1()
        self.block1_concat = Concat1()
        self.block1_dcl = DCL1()

        self.block2 = Block2()
        self.block2_eca = ECALayer2()
        self.block2_concat = Concat2()
        self.block2_dcl = DCL2()

        self.block3 = Block3()
        self.block3_eca = ECALayer3()
        self.block3_concat = Concat3()
        self.block3_dcl = DCL3()

        self.block4 = Block4()
        self.block4_eca = ECALayer4()
        self.block4_concat = Concat4()
        self.block4_dcl = DCL4()

        # self.PostBlock = PostBlock(self.num_classes)
        # self.margin = ArcMarginProduct(
        #     n_classes=self.num_classes, s=30, m=0.5, dtype="float32"
        # )

    def call(self, inputs):
        # inputs, y = inputs
        PreBlock = self.PreBlock(inputs)
        Block1 = self.block1(PreBlock)
        ECA1 = self.block1_eca(Block1)
        Concat1 = self.block1_concat(ECA1, Block1)
        DCL1 = self.block1_dcl(Concat1)

        Block2 = self.block2(DCL1)
        ECA2 = self.block2_eca(Block2)
        Concat2 = self.block2_concat(ECA2, Block2)
        DCL2 = self.block2_dcl(Concat2)

        Block3 = self.block3(DCL2)
        ECA3 = self.block3_eca(Block3)
        Concat3 = self.block3_concat(ECA3, Block3)
        DCL3 = self.block3_dcl(Concat3)

        Block4 = self.block4(DCL3)
        ECA4 = self.block4_eca(Block4)
        Concat4 = self.block4_concat(ECA4, Block4)
        DCL4 = self.block4_dcl(Concat4)

        output = DCL4
        # output = self.PostBlock(DCL4)
        # output = self.margin([output, y])
        return output
