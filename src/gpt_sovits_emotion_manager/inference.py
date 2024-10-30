import os
import re
import json
import random
import google.generativeai as genai
from typing import List, Optional, Literal
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from .log import log
from .api import generate
from .config import Config
from .utils import equal_emotions
from .models import EmotionAnnotation, Emotion


_prompt = """
你的任务是根据给定的文本生成对应的情绪标签。

情感标签由情感类型和强度组成。
情感类型包括: {emotion_types}
强度包括: low, moderate, high

请注意，情感标签可能不止一个，因此你需要为每个文本生成一个或多个情感标签。

你只需要返回一个包含情感标签的JSON数组，每个情感标签由情感类型和强度组成，不需要解释原因。

示例返回:
```json
[
    {"type": "joy", "intensity": "moderate"}
]
```

文本:
""".strip()


class Inferer:
    def __init__(
        self, emotion_annotations: List[EmotionAnnotation], config: Config
    ) -> None:
        """初始化推理器

        Args:
            emotion_annotations (List[EmotionAnnotation]): 情感标注对象列表
            config (Config): 配置对象
        """
        self.config = config
        self.emotion_annotations = emotion_annotations

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

    async def generate(
        self,
        text: str,
        language: Literal["zh", "ja", "en", "ko", "yue"],
        emotions: Optional[List[Emotion]] = None,
    ) -> bytes:
        """生成语音

        Args:
            text (str): 待合成的文本
            language (Literal[&quot;zh&quot;, &quot;ja&quot;, &quot;en&quot;, &quot;ko&quot;, &quot;yue&quot;]): 文本语言
            emotions (Optional[List[Emotion]], optional): 目标情感. Defaults to None.

        Returns:
            bytes: 生成的语音文件, 格式为 WAV
        """
        if emotions is None:
            log(
                "WARNING",
                f"No emotions specified, using default emotion: {self.config.emotion_types[0]}:low",
            )
            emotions = [Emotion(type=self.config.emotion_types[0], intensity="low")]

        emotion_annotations = self._find_emotion_annotations(emotions)
        ref_path = None
        aux_ref_path = []
        prompt_text = None
        prompt_language = None

        if len(emotion_annotations) == 0:
            log(
                "WARNING",
                f"No matched emotion annotation found, using the first emotion annotation: {self.emotion_annotations[0].file}",
            )
            ref_path = self.emotion_annotations[0].file
            prompt_text = self.emotion_annotations[0].text
            prompt_language = self.emotion_annotations[0].language
        elif len(emotion_annotations) == 1 and self.config.inference.use_aux_ref:
            log(
                "WARNING",
                f"Only one matched emotion annotation found, unable to use auxiliary reference",
            )
            ref_path = emotion_annotations[0].file
            prompt_text = emotion_annotations[0].text
            prompt_language = emotion_annotations[0].language
        else:
            ref_path = emotion_annotations[0].file
            aux_ref_path = [item.file for item in emotion_annotations[1:]]
            prompt_text = emotion_annotations[0].text
            prompt_language = emotion_annotations[0].language

        if len(aux_ref_path) > self.config.inference.max_aux_refs:
            aux_ref_path = random.sample(
                aux_ref_path, self.config.inference.max_aux_refs
            )

        if self.config.inference.use_aux_ref:
            log("INFO", f"Using {len(aux_ref_path)} auxiliary references")

        wav_file = await generate(
            base_url=self.config.inference.base_url,
            text=text,
            text_lang=language,
            ref_audio_path=ref_path,
            aux_ref_audio_paths=(
                aux_ref_path if self.config.inference.use_aux_ref else None
            ),
            prompt_text=prompt_text,
            prompt_lang=prompt_language,
            top_k=self.config.inference.top_k,
            top_p=self.config.inference.top_p,
            temperature=self.config.inference.temperature,
            text_split_method=self.config.inference.text_split_method,
            batch_size=self.config.inference.batch_size,
            batch_threshold=self.config.inference.batch_threshold,
            split_bucket=self.config.inference.split_bucket,
            speed_factor=self.config.inference.speed_factor,
            fragment_interval=self.config.inference.fragment_interval,
            streaming_mode=self.config.inference.streaming_mode,
            seed=self.config.inference.seed,
            parallel_infer=self.config.inference.parallel_infer,
            repetition_penalty=self.config.inference.repetition_penalty,
            media_type=self.config.inference.media_type,
        )

        return wav_file

    async def get_emotion_from_text(self, text: str) -> List[Emotion]:
        """使用 LLM 从文本中生成情感

        Args:
            text (str): 待分析的文本

        Returns:
            List[Emotion]: 从文本中生成的情感
        """
        response = await self.model.generate_content_async(
            _prompt.format(", ".join(self.config.emotion_types)) + text
        )
        json_str = re.sub(r"```json|```", "", response.text).strip()
        data = json.loads(json_str)
        emotions = []
        for item in data:
            if (
                "type" in item
                and "intensity" in item
                and item["type"]
                in {
                    "joy",
                    "fear",
                    "surprise",
                    "sadness",
                    "disgust",
                    "anger",
                    "neutral",
                    "confusion",
                }
                and item["intensity"] in {"low", "moderate", "high"}
            ):
                emotions.append(Emotion(**item))
        if not emotions:
            log("WARNING", "No emotion found in the text, using default emotion")
            emotions = [Emotion(type="neutral", intensity="low")]
        return emotions

    def _find_emotion_annotations(
        self, emotions: List[Emotion]
    ) -> List[EmotionAnnotation]:
        # 先找有没有完全匹配的
        matched_emotion_annotations = []
        for emotion_annotation in self.emotion_annotations:
            if equal_emotions(emotion_annotation.emotions, emotions):
                matched_emotion_annotations.append(emotion_annotation)

        if matched_emotion_annotations:
            return matched_emotion_annotations

        # 再找有没有部分匹配的
        intensity_mapping = {"low": 1, "moderate": 2, "high": 3}

        best_matches = []
        best_match_score = float("inf")

        for item in self.emotion_annotations:
            match_score = 0
            for target_emotion in emotions:
                found_match = False
                for emotion in item.emotions:
                    if emotion.type == target_emotion.type:
                        intensity_diff = abs(
                            intensity_mapping[emotion.intensity]
                            - intensity_mapping[target_emotion.intensity]
                        )
                        match_score += intensity_diff
                        found_match = True
                        break
                if not found_match:
                    # 如果目标情绪在项目中没有找到，则应用一个大的惩罚
                    match_score += 10

            if match_score < best_match_score:
                best_match_score = match_score
                best_matches = [item]
            else:
                best_matches.append(item)

        return best_matches
