#!/usr/bin/env python3
"""
Connectome Lab RAG Chatbot — Gradio-based web UI.
Supports Ollama (local), NVIDIA NIM, and NVIDIA Cloud backends.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import gradio as gr
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INDEX_DIR = PROJECT_ROOT / "rag" / "data" / "faiss_index"
CONFIG_PATH = PROJECT_ROOT / "rag" / "config" / "settings.json"

DEFAULT_CONFIG = {
    "mode": "ollama",
    "ollama_model": "qwen2.5:32b",
    "ollama_endpoint": "http://localhost:11434",
    "nvidia_api_key": "",
    "nvidia_model": "nemotron",
    "nim_endpoint": "http://localhost:8000/v1",
    "local_embed": "intfloat/e5-large-v2",
    "temperature": 0.3,
    "max_tokens": 1500,
    "top_k": 5,
}

SYSTEM_PROMPT = """당신은 서울대학교 뇌연결체 연구실(Connectome Lab, 지도교수: 차지욱)의 AI 연구 어시스턴트입니다.

## 역할
외부 연구자와 방문자가 본 연구실의 발표 논문을 정확히 이해할 수 있도록 돕습니다.

## 규칙
1. 반드시 제공된 논문 컨텍스트에 기반하여 답변하세요. 컨텍스트에 없는 내용은 추측하지 마세요.
2. 컨텍스트에 관련 정보가 없으면 "제공된 논문에서 해당 정보를 찾을 수 없습니다"라고 명확히 답하세요.
3. 논문을 인용할 때는 제목을 정확히 명시하세요. (예: "Machine learning prediction of incidence of Alzheimer's disease" 논문에 따르면...)
4. 한국어 질문에는 한국어로, 영어 질문에는 영어로 답변하세요.
5. 전문 용어는 원어를 괄호 안에 병기하세요. (예: 해마(hippocampus), 공포 일반화(fear generalization))
6. 답변은 구조적으로 작성하세요: 핵심 요약 → 세부 내용 → 관련 논문 목록.
7. 절대 URL이나 DOI를 지어내지 마세요. 확실한 정보만 제공하세요."""

CONTEXT_TEMPLATE = """아래는 서울대 뇌연결체 연구실(차지욱 교수)의 논문에서 검색된 관련 내용입니다.

{context}

---
위 논문 내용을 바탕으로 다음 질문에 답변해주세요.

질문: {query}"""


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            saved = json.load(f)
        config = {**DEFAULT_CONFIG, **saved}
    else:
        config = DEFAULT_CONFIG.copy()
    if os.environ.get("NVIDIA_API_KEY"):
        config["nvidia_api_key"] = os.environ["NVIDIA_API_KEY"]
    return config


def save_config(config: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


class RAGChatbot:
    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self.vectorstore = None
        self.retriever = None
        self._load_vectorstore()
        self._load_llm()

    def _load_vectorstore(self) -> None:
        if not INDEX_DIR.exists():
            logger.warning(f"FAISS index not found at {INDEX_DIR}. Run ingest.py first.")
            return

        from langchain_community.embeddings import HuggingFaceEmbeddings
        embedder = HuggingFaceEmbeddings(
            model_name=self.config["local_embed"],
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        from langchain_community.vectorstores import FAISS
        self.vectorstore = FAISS.load_local(
            str(INDEX_DIR), embedder, allow_dangerous_deserialization=True
        )
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.config["top_k"]},
        )
        logger.info("Vector store loaded successfully")

    def _load_llm(self) -> None:
        mode = self.config["mode"]
        if mode == "ollama":
            import requests
            try:
                r = requests.get(f"{self.config['ollama_endpoint']}/api/tags", timeout=5)
                models = [m["name"] for m in r.json().get("models", [])]
                logger.info(f"Ollama models available: {models}")
            except Exception as e:
                logger.warning(f"Ollama not reachable: {e}")
            logger.info(f"LLM: Ollama ({self.config['ollama_model']})")

        elif mode == "nvidia_nim":
            from openai import OpenAI
            self.nim_client = OpenAI(
                base_url=self.config["nim_endpoint"],
                api_key="not-needed",
            )
            logger.info(f"LLM: NVIDIA NIM ({self.config['nim_endpoint']})")

        elif mode == "nvidia_cloud":
            from openai import OpenAI
            self.cloud_client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=self.config["nvidia_api_key"],
            )
            logger.info(f"LLM: NVIDIA Cloud ({self.config['nvidia_model']})")

    def _translate_query(self, query: str) -> str:
        """Translate Korean query to English for better retrieval against English papers."""
        import re
        has_korean = bool(re.search(r'[\uac00-\ud7af\u1100-\u11ff\u3130-\u318f\ua960-\ua97f\ud7b0-\ud7ff]', query))
        if not has_korean:
            return query
        try:
            import requests
            resp = requests.post(
                f"{self.config['ollama_endpoint']}/api/chat",
                json={
                    "model": self.config["ollama_model"],
                    "messages": [
                        {"role": "system", "content": "Translate the following Korean neuroscience query into English. Output ONLY the English translation, nothing else."},
                        {"role": "user", "content": query},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 100},
                },
                timeout=30,
            )
            translated = resp.json()["message"]["content"].strip()
            logger.info(f"Query translated: '{query}' -> '{translated}'")
            return translated
        except Exception as e:
            logger.warning(f"Translation failed, using original query: {e}")
            return query

    def retrieve(self, query: str) -> list[dict]:
        if not self.retriever:
            return []
        en_query = self._translate_query(query)
        docs = self.retriever.invoke(en_query)
        seen = set()
        unique_docs = []
        for doc in docs:
            key = doc.page_content[:200]
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)

        return [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source_file", "unknown"),
                "page": doc.metadata.get("page", "?"),
                "title": doc.metadata.get("paper_title", "").replace("_", " "),
            }
            for doc in unique_docs
        ]

    def _format_context(self, docs: list[dict]) -> str:
        parts = []
        for i, d in enumerate(docs, 1):
            parts.append(
                f"### 논문 {i}: {d['title']}\n"
                f"(파일: {d['source']}, 페이지: {d['page']})\n\n"
                f"{d['content']}"
            )
        return "\n\n---\n\n".join(parts)

    def generate(self, query: str, context_docs: list[dict]) -> str:
        context = self._format_context(context_docs)
        user_message = CONTEXT_TEMPLATE.format(context=context, query=query)
        mode = self.config["mode"]

        if mode == "ollama":
            return self._generate_ollama(user_message)
        elif mode == "nvidia_nim":
            return self._generate_openai(self.nim_client, self.config["nvidia_model"], user_message)
        elif mode == "nvidia_cloud":
            return self._generate_openai(self.cloud_client, self.config["nvidia_model"], user_message)
        return "Error: Unknown mode"

    def _generate_ollama(self, user_message: str) -> str:
        import requests
        try:
            resp = requests.post(
                f"{self.config['ollama_endpoint']}/api/chat",
                json={
                    "model": self.config["ollama_model"],
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    "stream": False,
                    "options": {
                        "temperature": self.config["temperature"],
                        "num_predict": self.config["max_tokens"],
                    },
                },
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return f"LLM 오류: {e}"

    def _generate_openai(self, client, model: str, user_message: str) -> str:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=self.config["temperature"],
                max_tokens=self.config["max_tokens"],
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI-compatible API error: {e}")
            return f"LLM 오류: {e}"

    def chat(self, query: str) -> tuple[str, list[dict]]:
        if not query.strip():
            return "질문을 입력해주세요.", []
        context_docs = self.retrieve(query)
        if not context_docs:
            return (
                "관련 문서를 찾을 수 없습니다. FAISS 인덱스가 빌드되어 있는지 확인해주세요. "
                "(`python ingest.py` 실행 필요)",
                [],
            )
        answer = self.generate(query, context_docs)
        return answer, context_docs


def create_ui(chatbot: RAGChatbot) -> gr.Blocks:
    with gr.Blocks(title="SNU Connectome Lab — Research Chatbot") as app:
        gr.HTML("""
        <div style="text-align:center; padding:20px 0;">
            <h1 style="color:#1e3a5f; margin-bottom:4px;">SNU Connectome Lab</h1>
            <p style="color:#6b7280; font-size:14px;">서울대학교 뇌연결체 연구실 — AI Research Assistant</p>
            <p style="font-size:12px; color:#9ca3af;">Prof. Jiook Cha's Publications · Powered by Qwen2.5</p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=3):
                chatbox = gr.Chatbot(label="Chat", height=500, show_label=False)
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="연구실 논문에 대해 질문하세요... (예: '알츠하이머 예측에 어떤 방법을 사용했나요?')",
                        show_label=False,
                        scale=5,
                        container=False,
                    )
                    send_btn = gr.Button("전송", variant="primary", scale=1)

                gr.Examples(
                    examples=[
                        "연구실의 주요 연구 분야를 요약해주세요.",
                        "알츠하이머병 예측에 어떤 머신러닝 기법을 사용했나요?",
                        "fMRI Transformer 연구에 대해 설명해주세요.",
                        "공포 일반화(fear generalization) 연구 결과는?",
                        "해마(hippocampus)와 불안장애의 관계에 대한 연구를 알려주세요.",
                        "What neuroimaging techniques are used in the lab's studies?",
                    ],
                    inputs=msg_input,
                )

            with gr.Column(scale=1):
                gr.Markdown("### 참조 논문")
                sources_display = gr.HTML(
                    value="<p style='color:#9ca3af;'>질문 후 관련 논문이 여기에 표시됩니다.</p>"
                )
                with gr.Accordion("설정", open=False):
                    mode_selector = gr.Radio(
                        choices=["ollama", "nvidia_nim", "nvidia_cloud"],
                        value=chatbot.config["mode"],
                        label="추론 모드",
                    )
                    ollama_model_input = gr.Textbox(
                        value=chatbot.config.get("ollama_model", "qwen2.5:32b"),
                        label="Ollama 모델",
                    )
                    top_k_slider = gr.Slider(1, 10, value=chatbot.config["top_k"], step=1, label="Top-K 문서")
                    temp_slider = gr.Slider(0.0, 1.0, value=chatbot.config["temperature"], step=0.05, label="Temperature")

                gr.Markdown("---\n*[SNU Connectome Lab](https://connectomelab.snu.ac.kr)*")

        def respond(message: str, chat_history: list) -> tuple:
            answer, sources = chatbot.chat(message)
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": answer})

            if sources:
                html = ""
                for s in sources:
                    html += (
                        f'<div style="background:#f8fafc; border-left:3px solid #3b82f6; '
                        f'padding:8px 12px; margin:4px 0; border-radius:4px; font-size:13px;">'
                        f'<strong>{s["title"]}</strong><br>'
                        f'<span style="color:#6b7280;">p.{s["page"]} · {s["source"]}</span>'
                        f'</div>'
                    )
            else:
                html = "<p style='color:#9ca3af;'>검색된 논문이 없습니다.</p>"

            return chat_history, html, ""

        def update_settings(mode: str, ollama_model: str, top_k: int, temp: float) -> str:
            chatbot.config["mode"] = mode
            chatbot.config["ollama_model"] = ollama_model
            chatbot.config["top_k"] = top_k
            chatbot.config["temperature"] = temp
            save_config(chatbot.config)
            chatbot._load_llm()
            return f"설정 업데이트: mode={mode}, model={ollama_model}"

        send_btn.click(respond, [msg_input, chatbox], [chatbox, sources_display, msg_input])
        msg_input.submit(respond, [msg_input, chatbox], [chatbox, sources_display, msg_input])

        settings_output = gr.Textbox(visible=False)
        mode_selector.change(
            update_settings,
            [mode_selector, ollama_model_input, top_k_slider, temp_slider],
            settings_output,
        )

    return app


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Connectome Lab RAG Chatbot")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--mode", choices=["ollama", "nvidia_nim", "nvidia_cloud"], default=None)
    args = parser.parse_args()

    config = load_config()
    if args.mode:
        config["mode"] = args.mode

    chatbot = RAGChatbot(config)
    app = create_ui(chatbot)
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
