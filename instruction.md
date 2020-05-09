# 深度学习环境安装
---todo
### ubuntu系统安装
第一步是Linux系统的安装，在这里推荐使用Ubuntu16.04，[点此下载ubuntu系统][1]。当然你使用其他的系统版本也是可以的。操作系统的安装就不演示了。找一个u盘制作启动盘，几分钟就安装成功。
### CUDA 安装
>深度学习包含大量的矩阵计算，相比较CPU而言，拥有上千个处理器的GPU更适合用于深度学习。Nvidia开发的CUDA为我们提供了能够调用GPU来计算的接口。

1. [点击此处][2]确定你的显卡型号是否支持CUDA以及显卡的算力（compute capability)。
老旧的显卡算力会比较低，可能不支持最新版本的CUDA。[参考下表][3]，找到能够支持的CUDA版本：
![image_1cn10460s1668or31s4hpgik2p23.png-111kB][4]
2. 进入[CUDA Archive][5]，点击对应的版本，进入下载页。根据你的操作平台下载CUDA安装包（需要Nvidia Developer账号) 。
![image_1cn11e105npnqhm10b616ob2m12g.png-80.3kB][6]
3. [nouveau][7]是开源的Nvidia显卡驱动程序，为了成功安装官方驱动程序和CUDA，执行```lsmod | grep nouveau```，如果显示了一些nouveau的相关信息，按如下步骤关闭nouveau。
```
$ sudo touch /etc/modprobe.d/blacklist-nouveau.conf # 创建文件
$ sudo vim /etc/modprobe.d/blacklist-nouveau.conf # 进入文字编辑模型 
# 按i，然后输入：
blacklist nouveau
options nouveau modeset=0
# 按esc，再按：，输入wq，回车，保存退出。
# 执行：
$ sudo update-initramfs -u
```
再执行```lsmod | grep nouveau```，如果没有任何显示，则关闭成功，否则重启即可。
4. 按 CRTL + ALT + F1进入终端登录，输入用户名和密码。
```
# 执行如下命令关闭图形界面
$ sudo service lightdm stop
# 进入cuda安装包的位置，默认为~/Downloads
$ cd ~/Downloads 
$ sudo sh cuda_<version>_linux.run
```
进入到CUDA的安装界面，除了不安装OpenGL，按照提示，一路接受即可。
5. 执行如下命令，重新开启图形界面。
```
sudo service lightdm start
```
按CRTL + ALT + F7进入图形界面登录。
6. 安装完毕，这时，我们的电脑上安装了3个东西：1) 官方驱动; 2) CUDA库; 3)　CUDA示例代码。执行```nvidia-smi```，检查显卡和驱动信息。
7. 为了让系统能够找到CUDA的位置，我们需要设置系统环境变量
``` 
$ sudo vim ~/.bashrc
# 按i，然后输入：
export PATH=/usr/local/cuda-9.2/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-9.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
# 按esc，再按：，输入wq，回车，保存退出。
# 执行：
$ source ~/.bashrc　# 让环境变量立即生效
```
8.接下来使用刚刚安装好的CUDA代码样例检查CUDA是否能够正常运行。
```
# 进入CUDA代码样例，默认为~/NVIDIA_CUDA-9.2_Samples。
$ cd ~/NVIDIA_CUDA-9.2_Samples/1_Utilities/deviceQuery
$ make # 编译代码
$ ./deviceQuery # 执行刚刚编译出的可执行程序
```
如果出现```Result = PASS```，则运行成功，自行对照显卡、驱动、CUDA等信息。
9. 参考[英伟达官方CUDA安装指导][8] 

### CUDNN 安装
> CUDA提供了利用GPU进行数学计算的接口，而CUDNN则专门提供了神经网络计算的加速接口。

1. cudnn的安装十分简单，进入[cudnn Archive][9]，选择对应的版本下载。
2. 进入下载路径, 执行如下命令：
```
$ tar -xzvf cudnn-9.2-linux-x64-v7.2.1.38.tgz # 解压cudnn压缩包
$ sudo cp cuda/include/cudnn.h /usr/local/cuda/include # 复制头文件
$ sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64 # 复制库文件
$ sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn* # 更改文件权限
```
3. 参考[英伟达官方CUDNN安装指导][10]

###　Anaconda 安装
> Anaconda是Python的科学计算发行版，包含了非常多的库，依赖和工具。conda是相关的环境安装、管理工具。

1. 首先下载Anaconda安装包，推荐到[清华镜像站][11]下载。
2. 到下载路径，执行```$ bash Anaconda3-5.2.0-Linux-x86_64.sh```，根据提示，安装即可。
3. 配置anaconda的源，比如[清华镜像站][12]：
```
$ conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
$ conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
$ conda config --set show_channel_urls yes
```

### OpenCV 安装
> OpenCV是开源的计算机视觉库，它包含了非常多的图像，视频处理接口，随着计算机视觉的不断发展，它也在不断的包含许多优秀的开源实现。同时，它也是一些深度学习框架的依赖。

在这里我们提供两种安装方式，源码编译安装和使用pip安装OpenCV库。
```
$ git clone https://github.com/opencv/opencv.git # 下载OpenCV源码和git项目
$ git clone https://github.com/opencv/opencv_contrib.git # opencv_contrib包含了一些有版权的项目和正在开发的项目，下载
$ cd opencv # 进入OpenCV源码根目录
$ git checkout 3.4.2 # 将源码切换到3.4.2版本
$ mkdir build # 新建一个build文件夹，用于存储编译内容
$ cd build
$ cmake \
-DOPENCV_EXTRA_MOUDULE_PATH=~/opencv_contrib/modules \
-DCMAKE_ISNTALL_PREFIX=/usr/local \
-DPYTHON_EXECUTABLE=$(which python) \
-DPYTHON_INCLUDE_DIR= $(python -c "from distutils.sysconfig import get_python_inc;print(get_python_inc())") \
-DPYTHON_PACKAGES_PATH=$(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
 -DWITH_IPP=OFF \
 -DWITH_CUDA=OFF \
 ..
``` 
其中，-D为在编译的时候定义宏，OPENCV_EXTRA_MOUDULE_PATH为指定opencv_contrib的路径，CMAKE_ISNTALL_PREFIX为指定make install的安装路径。与Python有关的3行表示使用指定的python。关闭了IPP加速和CUDA部分，为了加快cmake的过程和编译速度。
接着执行如下过程编译和安装库文件。
``` 
$ make -j8　#-j表示用几个核去编译
$ sudo make install # 安装库文件到系统中
```
这样，OpenCV就安装成功了。
如果需要在Python中使用OpenCV，推荐使用pip直接安装：
```
$ pip install opencv-python
$ pip install opencv-contrib-python
```
### Darknet 安装
> Darknet是使用C语言编写的深度学习框架，只依赖于linux平台。Darknet十分轻量，安装方便，YOLO目标检测来自于此。
```
$ git clone https://github.com/pjreddie/darknet.git
$ cd darknet
$ vim Makefile #　修改Makefile，打开GPU和OpenCV的开关
$ make #　编译
$ wget https://pjreddie.com/media/files/yolov3.weights　# 下载YOLO3的模型
$ ./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg　# 测试yolo
```
### caffe 安装
> caffe是使用C++编写深度学习框架，提供C++和Python接口。比较适用于计算机视觉。

1.因为caffe只支持编译安装，而且依赖较多，安装过程比较繁琐，我们先下载caffe的代码。
```
$ git clone https://github.com/BVLC/caffe.git
```
2.安装各种依赖库：
```
$ sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
$ sudo apt-get install --no-install-recommends libboost-all-dev
$ sudo apt-get install libatlas-base-dev
$ sudo apt-get install libopenblas-dev
$ sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev
```

3.建立一个conda虚拟环境。
```
$ conda create -n pycaffe python=3.5　# 建立一个python版本为3.5，环境名为pycaffe的虚拟环境
$ source activate pycaffe　# 进入虚拟环境
```
4.按照你是使用Python3还是Python2，修改CMakeLists.txt
```
# 修改
set(python_version "2" CACHE STRING "Specify which Python version to use")
# 为
set(python_version "3" CACHE STRING "Specify which Python version to use")
```
5.按照如下流程编译代码和测试。
```
$ cd caffe
$ mkdir build
$ cd build
$ cmake ..
```
cmake完成之后对应每一行cmake信息，检查相关依赖是否被找到，以及路径是否正确。比如：
![image.png-82.8kB][13]
接着编译代码
```
$ make all
$ make install
$ make runtest
```
6.在虚拟环境中安装pycaffe的依赖
```
$ cd ~/caffe
$ for req in $(cat requirements.txt); do pip install $req; done
```
7.将pycaffe所在路径添加到PYTHONPATH环境变量当中。方便python能够找到pycaffe
```
$ vim ~/.bashrc
# 添加下面这行到~/.bashrc当中，注意修改路径
PYTHONPATH=/path/to/caffe/python:$PYTHONPATH
```
8.进入python交互式环境，测试caffe是否安装成功。
```
>>> import caffe
>>> caffe.set_mode_gpu() # 设置为gpu模式
```
注意：如果在```import caffe```发生matplotlib相关的报错信息，参考[此处解决][14]。
9. 然后执行```$ nvidia-smi```查看gpu是否有相关进程在占用，如：
![image.png-58.5kB][15]

### pytorch和tensorflow的安装
推荐使用pip或conda直接安装库文件，十分简单，但注意使用虚拟环境。这里就不赘述了，各自主页[pytorch][16][tensorflow][17]都提供针对不同平台的安装方法，执行命令即可。


  [1]: http://releases.ubuntu.com/16.04/ubuntu-16.04.5-desktop-amd64.iso.torrent?_ga=2.142199028.770989430.1536549700-551296579.1536549700
  [2]: https://developer.nvidia.com/cuda-gpus
  [3]: https://en.wikipedia.org/wiki/CUDA
  [4]: http://static.zybuluo.com/guanjiexiong/8wa4du5xp5rq35d8crnmnhjz/image_1cn10460s1668or31s4hpgik2p23.png
  [5]: https://developer.nvidia.com/cuda-toolkit-archive
  [6]: http://static.zybuluo.com/guanjiexiong/iqqizvxp74lhn09j9sm6ycan/image_1cn11e105npnqhm10b616ob2m12g.png
  [7]: https://en.wikipedia.org/wiki/Nouveau_%28software%29
  [8]: https://developer.download.nvidia.com/compute/cuda/9.2/Prod2/docs/sidebar/CUDA_Installation_Guide_Linux.pdf
  [9]: https://developer.nvidia.com/rdp/cudnn-archive
  [10]: https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html
  [11]: https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/
  [12]: https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/
  [13]: http://static.zybuluo.com/guanjiexiong/5yyrr8e1sbrzoxb0elztq7gt/image.png
  [14]: https://stackoverflow.com/questions/32079919/trouble-with-importing-matplotlib
  [15]: http://static.zybuluo.com/guanjiexiong/wz94gmb81ajkej0eq09cu89t/image.png
  [16]: https://pytorch.org/
  [17]: https://www.tensorflow.org/install/
