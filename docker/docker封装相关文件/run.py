#! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import os
import pandas as pd
import testmodel



if __name__ == "__main__":

    #################### 不可修改区域开始 ######################
    testpath = '/home/data/'						#测试集路径。包含验证码图片文件
    result_folder_path = '/result/submission.csv'	#结果输出文件路径
	#################### 不可修改区域结束 ######################


	### 调用自己的工程文件，并这里生成结果文件（datafram）
    result = testmodel.model(testpath)
    print(result)
	# 注意路径不能更改，index需要设置为None
    result.to_csv(result_folder_path, index=None)
	### 参考代码结束：输出标准结果文件
