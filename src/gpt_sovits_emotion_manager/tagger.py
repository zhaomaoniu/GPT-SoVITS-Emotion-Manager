import re
import os
import json
import asyncio
from typing import List
from pathlib import Path
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from .log import log
from .config import Config
from .utils import get_audio_duration
from .models import ListFileAnnotation, EmotionAnnotation, Emotion


_prompt = """
你的任务是为每个文本进行情感标注。请根据文本的内容，选择最符合的情感标签。

文本格式: 符合下列规则的多行文本
(标识符)角色名: 文本内容

输出格式:
```json
{
    "标识符1": [
        {
            "type": "情感标签, 可选项有: {emotion_types}",
            "intensity": "情感强度, 可选项有: `low`, `moderate`, `high`"
        },
        ...
    ],
    "标识符2": [
        {
            "type": "情感标签",
            "intensity": "情感强度"
        },
        ...
    ],
    ...
}
```

请注意:
- 每行至多有一个标识符。有些行数可能没有标识符或角色名，只有文本内容，这是不需要进行标注的内容，是用于辅助你判断情感标签的。
- 文本有可能构成上下文，在这种情况下，你可以根据上下文来更准确地判断情感标签，否则请仅根据单行文本内容来判断。
- 在单一情感无法准确标注时，请组合多种情感类型。
- 请确保每个标识符都有对应的情感标签。

文本：
""".strip()


class Tagger:
    def __init__(self, config: Config) -> None:
        """初始化情感标注器

        Args:
            config (Config): 配置对象
        """
        self.config = config
        genai.configure(api_key=config.llm.api_key)

        if config.llm.proxy:
            # 因为 `google.genrativeai` 底层使用 `gRPC` 通信，所以只能通过环境变量设置代理
            os.environ["HTTP_PROXY"] = config.llm.proxy
            os.environ["HTTPS_PROXY"] = config.llm.proxy

        self.model = genai.GenerativeModel(
            model_name=config.llm.model,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
            generation_config={"max_output_tokens": int(1e6)},
        )

    def from_list_file(self, list_file_path: str) -> List[ListFileAnnotation]:
        """从列表文件中读取标注信息

        Args:
            list_file_path (str): 列表文件路径

        Returns:
            List[ListFileAnnotation]: 标注列表
        """
        with open(list_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        annotations = []
        for line in lines:
            file, speaker, language, text = line.strip().split("|")
            annotations.append(
                ListFileAnnotation(
                    path=file, speaker=speaker, language=language.lower(), text=text
                )
            )

        return annotations

    def _generate_input(self, list_file_annotation: ListFileAnnotation) -> str:
        if (
            list_file_annotation.path == "null"
            and list_file_annotation.speaker == "null"
        ):
            return list_file_annotation.text

        if list_file_annotation.path == "null":
            return f"{list_file_annotation.speaker}: {list_file_annotation.text}"

        if list_file_annotation.speaker == "null":
            return (
                f"({Path(list_file_annotation.path).stem}) {list_file_annotation.text}"
            )

        return f"({Path(list_file_annotation.path).stem}){list_file_annotation.speaker}: {list_file_annotation.text}"

    async def tag(
        self, list_file_annotation: List[ListFileAnnotation], retry: int = 5
    ) -> List[EmotionAnnotation]:
        """进行情感标注

        Args:
            list_file_annotation (List[ListFileAnnotation]): 使用 `from_list_file` 方法生成的标注列表

        Returns:
            List[EmotionAnnotation]: 情感标注列表
        """
        redundancy = 20

        async def process_batch(
            batch: List[ListFileAnnotation], retry: int
        ) -> List[EmotionAnnotation]:
            cnt = 0
            while cnt < retry:
                try:
                    return await self._tag(batch)
                except Exception as e:
                    cnt += 1
                    log(
                        "WARNING",
                        f"Occurred error, retrying(<c>{cnt}/{retry}</c>): {e}",
                    )
            log(
                "ERROR",
                f"Failed to process batch with start line: <b>{batch[0].text}</b>",
            )
            prompt = _prompt + "\n".join([self._generate_input(item) for item in batch])
            log("DEBUG", f"Prompt: {prompt}\n")
            return []

        # 分批处理
        tasks = [
            process_batch(list_file_annotation[i : i + 200 + redundancy], retry)
            for i in range(0, len(list_file_annotation), 200)
        ]
        # 使用 asyncio.gather 并行运行所有任务
        all_results = await asyncio.gather(*tasks)

        # 合并结果并去重
        files = set()
        result_filtered = []
        for batch_result in all_results:
            for r in batch_result:
                if r.file not in files:
                    files.add(r.file)
                    result_filtered.append(r)

        return result_filtered

    async def _tag(
        self, list_file_annotation: List[ListFileAnnotation]
    ) -> List[EmotionAnnotation]:
        """使用 Gemini 进行情感标注

        Args:
            list_file_annotation (List[ListFileAnnotation]): 使用 `from_list_file` 方法生成的标注列表

        Raises:
            ValueError: 如果 Gemini 返回的数据不符合预期
            json.JSONDecodeError: 如果 Gemini 返回的数据无法解析

        Returns:
            List[EmotionAnnotation]: 情感标注列表
        """
        prompt = (
            _prompt.format(emotion_types=", ".join(self.config.emotion_types))
            + "\n"
            + "\n".join([self._generate_input(a) for a in list_file_annotation])
        )

        log(
            "INFO",
            f"Requesting Gemini:\n<dim>{prompt[len(_prompt): len(_prompt) + 100].strip()}...</dim>",
        )

        result = await self.model.generate_content_async(prompt)
        text = result.text
        # 这里可能 ValueError，记得在外面处理

        text = re.sub(r"```json|```", "", text).strip()

        annotations = []
        data = json.loads(text)
        # 这里可能 json.JSONDecodeError，记得在外面处理

        for a in list_file_annotation[:200]:
            # 留有 20 的冗余，防止 Gemini 返回的数据不完整
            if a.path == "null":
                continue

            file = Path(a.path).stem

            if file not in data:
                raise ValueError(f"Invalid file name: {file}")

            emotions = []
            for emotion in data[file]:
                if emotion["type"] not in self.config.emotion_types:
                    raise ValueError(f"Invalid emotion type: {emotion}")
                if emotion["intensity"] not in ["low", "moderate", "high"]:
                    raise ValueError(f"Invalid emotion intensity: {emotion}")

                emotions.append(
                    Emotion(
                        type=emotion["type"],
                        intensity=emotion["intensity"],
                    )
                )

            annotations.append(
                EmotionAnnotation(
                    file=a.path,
                    text=a.text,
                    language=a.language,
                    emotions=emotions,
                )
            )
        return annotations

    def check_duration(
        self, annotations: List[EmotionAnnotation]
    ) -> List[EmotionAnnotation]:
        """检查音频时长

        Args:
            annotations (List[EmotionAnnotation]): 情感标注列表

        Returns:
            List[EmotionAnnotation]: 情感标注列表
        """
        return [a for a in annotations if 3 <= get_audio_duration(a.file) <= 10]
