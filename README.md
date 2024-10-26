# GPT-SoVITS-Emotion-Manager

这是一个适用于 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 的情感控制方案，通过使用 LLM（Language Level Model）对 .list 文件进行标注，进行了对参考音频的分类，进而实现了对情感的控制。

## 功能

- [x] 通过 LLM 对 .list 文件进行标注，生成一份情感标注文件
- [x] 通过情感标注文件，实现自动化的参考音频选择，即情感控制
- [x] 调用 GPT-SoVITS/api_v2.py 提供的 API 进行音频生成
- [ ] 支持除 Gemini 以外的其他 LLM 模型
- [ ] 支持自定义情感类型

## 使用

1. 克隆本仓库

```bash
git clone https://github.com/zhaomaoniu/GPT-SoVITS-Emotion-Manager.git
```

2. 安装依赖
<details open>
<summary>使用 PDM 安装</summary>
在项目的根目录下打开命令行, 输入以下指令即可安装

    pdm install

</details>

<details>
<summary>使用其他包管理器安装</summary>
在项目的插件目录下, 打开命令行, 根据你使用的包管理器, 输入相应的安装命令

<details>
<summary>pip</summary>

    pip install -r requirements.txt
</details>
<details>
<summary>poetry</summary>

    poetry install
</details>
<details>
<summary>conda</summary>

    conda install --file requirements.txt
</details>

</details>

3. 运行 Tagger

```bash
pdm run run_tagger.py -f <list_file>
```

这会在 outputs/emotions 目录下生成一个情感标注文件，你可以打开这个文件查看标注结果并手动校准。

4. 运行 Inferer

```bash
pdm run run_inferer.py -f <emotion_file>
```

`run_inferer.py` 使用命令行进行交互，你可以输入文本，情感，语言来生成对应的音频。特别地，在不输入情感的情况下，程序会调用 LLM 进行情感识别。

## 配置

你需要在 `config.yaml` 中配置一些参数，以保证程序正常运行。

配置文件内的注释很详细，推荐直接查看配置文件修改。

在一般情况下，你只修改配置 `llm` 中的 `api_key`。它可以在 [Google AI Studio](https://aistudio.google.com/app/apikey) 中获取。

## 感谢

- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS): 优秀的 TTS 方案
- [ムラサメ](https://www.yuzu-soft.com/products/senren/chara.html): 本项目的编写的直接原因