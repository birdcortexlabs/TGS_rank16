文件说明与运行顺序：

feat部分中的jigsawpuzzle.py,jigsawpuzzle2.py用来生成拼图

model文件夹中covernet.py,depthnet.py,emptynet2.py用来生成特征文件，可以直接运行

inception.py,unet5_bn.py,unet8_bn.py,unet10_bn.py,unet11_bn.py生成3个5折模型，2个10折模型。运行方式python inception.py <num of folds>

stacking可供参考，最后没用这种融合方式。

merge4.py是最后提交的融合方式

feat部分中的jigsawpuzzle3.py用来生成最后结果的拼图，用来debug规则