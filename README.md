# 2025年秋季国科大《GPU架构与编程》作业二
# https://github.com/Zymonody7/musa_inference
本项目是一个极简的大模型推理服务模板，旨在帮助您快速构建一个可以通过API调用的推理服务器。


## 项目结构

- `FakeDockerfile`: 用于构建的配置文件,但是这个是假的，因为摩尔线程的评测机是容器实例，没有办法再创建容器了，这里的FROM并不是拉取容器镜像。**请不要修改此文件的 EXPOSE 端口和 CMD 命令**。
- `serve.py`: 推理服务的核心代码。您需要在此文件中修改和优化您的模型加载与推理逻辑。这个程序不能访问Internet。
- `requirements.txt`: Python依赖列表。您可以添加您需要的库。
- `.gitignore`: Git版本控制忽略的文件列表。
- `download_model.py`: 下载权重的脚本，可以自行修改，请确保中国大陆的网络能够下载到。可以使用阿里云对象存储等云平台，或者参考沐曦代码模板中的方式。
- `README.md`: 本说明文档。

## 如何修改

您需要关注的核心文件是 `serve.py`。

目前，它使用 `transformers` 库加载了一个非常小的模型 `Qwen/Qwen2.5-0.5B`。您可以您可以完全替换 `serve.py` 的内容，只要保证容器运行后，能提供模板中的'/predict'和'/'等端点即可。

**重要**: 评测系统会向 `/predict` 端点发送 `POST` 请求，其JSON body格式为：

```json
{
  "prompt": "Your question here"
}

您的服务必须能够正确处理此请求，并返回一个JSON格式的响应，格式为：

```json
{
  "response": "Your model's answer here"
}
```

**请务必保持此API契约不变！**

## 环境说明

### 软件包版本

Driver version: 2.7.0
MUSA version: 3.1.0

项目的基础环境是这样的，不要替换任何和torch相关的包，否则无法调用GPU，这个模板的工作方式是直接复制一份具有下列软件包的conda环境然后再用pip安装其它的包。

Package                        Version
------------------------------ ------------------
absl-py                        2.2.2
anyio                          4.9.0
argon2-cffi                    23.1.0
argon2-cffi-bindings           21.2.0
arrow                          1.3.0
asttokens                      3.0.0
async-lru                      2.0.5
attrs                          25.3.0
babel                          2.17.0
beautifulsoup4                 4.13.3
bleach                         6.2.0
blinker                        1.9.0
brotlipy                       0.7.0
certifi                        2022.12.7
cffi                           1.15.1
charset-normalizer             2.0.4
click                          8.3.1
comm                           0.2.2
conda                          22.11.1
conda-content-trust            0.1.3
conda-package-handling         1.9.0
contourpy                      1.3.1
cryptography                   38.0.1
cycler                         0.12.1
debugpy                        1.8.14
decorator                      5.2.1
defusedxml                     0.7.1
docker                         7.1.0
exceptiongroup                 1.2.2
executing                      2.2.0
fastjsonschema                 2.21.1
filelock                       3.18.0
Flask                          3.1.2
fonttools                      4.57.0
fqdn                           1.5.1
fsspec                         2025.3.2
gitdb                          4.0.12
GitPython                      3.1.45
grpcio                         1.71.0
h11                            0.14.0
httpcore                       1.0.8
httpx                          0.28.1
idna                           3.4
ipykernel                      6.29.5
ipython                        8.35.0
ipywidgets                     8.1.6
isoduration                    20.11.0
itsdangerous                   2.2.0
jedi                           0.19.2
Jinja2                         3.1.6
joblib                         1.5.2
json5                          0.12.0
jsonpointer                    3.0.0
jsonschema                     4.23.0
jsonschema-specifications      2024.10.1
jupyter_client                 8.6.3
jupyter_core                   5.7.2
jupyter-events                 0.12.0
jupyter-lsp                    2.2.5
jupyter_server                 2.15.0
jupyter_server_terminals       0.5.3
jupyterlab                     4.4.0
jupyterlab-language-pack-zh-CN 4.3.post3
jupyterlab_pygments            0.3.0
jupyterlab_server              2.27.3
jupyterlab_widgets             3.0.14
kiwisolver                     1.4.8
Markdown                       3.8
MarkupSafe                     3.0.2
matplotlib                     3.10.1
matplotlib-inline              0.1.7
mistune                        3.1.3
mpmath                         1.3.0
nbclient                       0.10.2
nbconvert                      7.16.6
nbformat                       5.10.4
nest-asyncio                   1.6.0
networkx                       3.4.2
nltk                           3.9.2
notebook_shim                  0.2.4
numpy                          1.26.4
overrides                      7.7.0
packaging                      24.2
pandocfilters                  1.5.1
parso                          0.8.4
pexpect                        4.9.0
pillow                         11.2.1
pip                            22.3.1
platformdirs                   4.3.7
pluggy                         1.0.0
prometheus_client              0.21.1
prompt_toolkit                 3.0.50
protobuf                       6.30.2
psutil                         7.0.0
ptyprocess                     0.7.0
pure_eval                      0.2.3
pycosat                        0.6.4
pycparser                      2.21
Pygments                       2.19.1
pyOpenSSL                      22.0.0
pyparsing                      3.2.3
PySocks                        1.7.1
python-dateutil                2.9.0.post0
python-json-logger             3.3.0
PyYAML                         6.0.2
pyzmq                          26.4.0
referencing                    0.36.2
regex                          2025.11.3
requests                       2.32.3
rfc3339-validator              0.1.4
rfc3986-validator              0.1.1
rouge-score                    0.1.2
rpds-py                        0.24.0
ruamel.yaml                    0.17.21
ruamel.yaml.clib               0.2.6
scipy                          1.15.2
Send2Trash                     1.8.3
setuptools                     65.5.0
six                            1.16.0
smmap                          5.0.2
sniffio                        1.3.1
soupsieve                      2.6
stack-data                     0.6.3
sympy                          1.13.3
tensorboard                    2.19.0
tensorboard-data-server        0.7.2
terminado                      0.18.1
tinycss2                       1.4.0
tomli                          2.2.1
toolz                          0.12.0
torch                          2.2.0a0+git8ac9b20
torch_musa                     1.3.0
torchvision                    0.17.2+c1d70fe
tornado                        6.4.2
tqdm                           4.64.1
traitlets                      5.14.3
types-python-dateutil          2.9.0.20241206
typing_extensions              4.13.2
uri-template                   1.3.0
urllib3                        1.26.13
wcwidth                        0.2.13
webcolors                      24.11.1
webencodings                   0.5.1
websocket-client               1.8.0
Werkzeug                       3.1.3
wheel                          0.37.1
widgetsnbextension             4.0.14


### judge平台的配置说明

judge机器的配置如下：

``` text
os: ubuntu22.04
cpu: 15核
内存: 100GB
磁盘: 50GB（已占用1.9GB，剩下的可用，将权重存储到相对路径是最安全的）
GPU: MTT S4000(显存：48GB)
网络带宽：1-2Gbps，这个网络是区域的总带宽，不要太依赖它
```

judge系统的配置如下：

``` text
docker build stage: 900s
docker run - health check stage: 180s
docker run - predict stage: 360s
```


