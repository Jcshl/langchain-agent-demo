# Dify安装教程

## 一、Dify简介:

- Dify 是一款开源的大语言模型（LLM）应用开发平台，使开发者可以快速搭建生产级的生成式 AI 应用。即使你是非技术人员，也能参与到 AI 应用的定义和数据运营过程中。

  ![image-20260118173832473](img/01.png)

## 二、Dify安装教程（Windows 系统安装步骤）

## 1.1 下载 Docker Desktop

- 下载地址：https://docs.docker.com/desktop/setup/install/windows-install/

![image-20260207113110823](img/05.png)

## 1.2 系统要求

- 注意：有两种方式：WSL或者Hyper-V.Hyper-V 和 WSL 各有优缺点，具体取决于您的配置和使用场景。

### 1.3 基于WSL（默认方式）

![1773029588578](img/18.png)

#### 1.3.1启用虚拟机管理程序：

- 打开命令：控制面板--》程序--》启用或者关闭windows功能

![1773029980944](img/20.png)

#### 1.3.2 WSL 验证和设置

请先通过在终端中运行以下命令来验证您安装的版本是否满足系统要求：win+R

```console
wsl --version
```

如果未显示版本详细信息，则您可能正在使用 WSL 的默认版本。此版本不支持最新功能，必须进行更新。

您可以使用以下方法之一更新或安装 WSL：

通过终端安装或更新 WSL

1. 以管理员身份打开 PowerShell 或 Windows 命令提示符。
2. 运行安装或更新命令。系统可能会提示您重启计算机。更多信息，请参阅

```console
wsl --install

wsl --update
```

![1773036716242](img/22.png)

#### 1.3.3 安装docker

默认安装即可

### 1.4 基于Hyper-V（可选方式）

![1773029588578](img/19.png)

#### 1.4.1 开启windows的Hyper-V虚拟化技术

- 打开控制面板——>程序——>启动或关闭window功能


![image-20260207113345029](img/06.png)

![image-20260207113405501](img/07.png)

- 需要开启Hyper-V虚拟化技术。如果您的计算机不支持Hyper-V或者没有启用Hyper-V，则会出现Windows无法启动Docker的问题。
  ![image-20260207113531128](img/08.png)
#### 1.4.2 安装Docker

- 双击运行下载好的安装包

  ![image-20260207113756843](img/09.png)

- 注意：有的同学可能会出现以下报错： Docker Desktop requires Windows 10 Pro/Enterprise/Home version 19044 or above.意思是docker只支持 Windows 10 专业版/企业版/家庭版，并且版本要高于19044好吧，到这里我都想升级windows系统或者重装系统了

- 大家如果没有出现这种报错就不用管了。操作系统版本修改方式如下：

- 输入cmd命令，运行regedit，回车，出现[注册表]编辑器，找到：计算机\HKEY LOCAL MACHINE\SOFTWARE\Microsoft\Windows NT\CurrentVersion。确认EditionId是否为Professional，如果不是对其进行修改。

  ![image-20260207114916214](img/10.png)

- 修改CurrentBuild和CurrentBuildNumber，改为提示要求的19044

  ![image-20260207115003558](img/11.png)

- 然后重新双击安装，进入到了安装界面，注意，不要勾选第一项，不然会安装失败，因为我们设置了Hyper-v。

  ![image-20260207115029005](img/12.png)

- 点击close and log out：会重启电脑，重启后正常打开就可以了

  ![image-20260207115053159](img/13.png)



### 1.5 打开 Docker：

- 不用登录，直接点击不登录

  ![image-20260207130239097](img/14.png)

### 1.6 进入Desktop Docker

![image-20260207130321759](img/15.png)

### 1.7 设置资源下载位置

- 默认是在C盘，根据自己需求进行修改。

  ![image-20260207130417496](img/16.png)

### 1.8 设置镜像

- 不然无法下载镜像，配置如下

  ```json
  {
    "builder": {
      "gc": {
        "defaultKeepStorage": "20GB",
        "enabled": true
      }
    },
    "experimental": false,
    "features": {
      "buildkit": true
    },
    "registry-mirrors": [
      "https://docker.1panelproxy.com",
      "https://2a6bf1988cb6428c877f723ec7530dbc.mirror.swr.myhuaweicloud.com",
      "https://docker.m.daocloud.io",
      "https://hub-mirror.c.163.com",
      "https://mirror.baidubce.com",
      "https://your_preferred_mirror",
      "https://dockerhub.icu",
      "https://docker.registry.cyou",
      "https://docker-cf.registry.cyou",
      "https://dockercf.jsdelivr.fyi",
      "https://docker.jsdelivr.fyi",
      "https://dockertest.jsdelivr.fyi",
      "https://mirror.aliyuncs.com",
      "https://dockerproxy.com",
      "https://mirror.baidubce.com",
      "https://docker.m.daocloud.io",
      "https://docker.nju.edu.cn",
      "https://docker.mirrors.sjtug.sjtu.edu.cn",
      "https://docker.mirrors.ustc.edu.cn",
      "https://mirror.iscas.ac.cn",
      "https://docker.rainbond.cc"
    ]
  }
  ```

  ![1773025742153](img/17.png)

### 1.9 验证docker

- 打开cmd，执行下面命令

```
docker --version
```

- 拉取镜像操作

  ```bash
  docker pull hello-world
  ```

## 2 克隆 Dify 仓库

- 输入下面命令

  ```bash
  git clone https://github.com/langgenius/dify.git
  ```

- > 如果无法clone，可以直接download原始的zip文件（解压缩）：下载地址https://github.com/langgenius/dify.git，访问 GitHub 仓库页面 → 点击 "Code" → 选择 "Download ZIP"

#### 注意：如果无法下载，请直接使用配套资料中的"dify"文件夹即可

## 3 进入 Docker 目录（环境配置）
  - 进入到dify文件夹下docker文件夹，将.env.example文件名改为 .env

    > 切换到dify/docker文件夹下去操作，可以使用rename .env.example .env


- 也可以输入下面命令：


  ```bash
  cd dify/docker 
  cp .env.example .env
  ```

## 4 启动 Dify

- 先进入dify文件找到docker文件夹，然后进入终端，再去操作下面的命令

- 输入下面命令：

  ```bash
  docker compose up -d
  ```

![image-20260118174928034](img/03.png)

## 5 访问应用

- 登陆：http://localhost/signin；设置好邮箱、用户名、密码就可以登陆了

![image-20260118175105414](img/04.png)