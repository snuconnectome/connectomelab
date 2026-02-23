#!/usr/bin/env python3
"""
Connectome Lab RAG Chatbot — Gradio-based web UI.
Supports both NVIDIA NIM (cloud/local) and local HuggingFace models.
Minimum latency design with streaming responses.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Generator, Optional

import gradio as gr
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INDEX_DIR = PROJECT_ROOT / "rag" / "data" / "faiss_index"
CONFIG_PATH = PROJECT_ROOT / "rag" / "config" / "settings.json"

DEFAULT_CONFIG = {
    "mode": "local",  # "local" | "nvidia_cloud" | "nvidia_nim"
    "nvidia_api_key": "",
    "nvidia_model": "nvidia/nemotron-3-nano-30b-a3b",
    "nvidia_embed_model": "nvidia/nv-embedqa-e5-v5",
    "local_llm": "microsoft/Phi-3.5-mini-instruct",
    "local_embed": "intfloat/e5-large-v2",
    "nim_endpoint": "http://localhost:8000/v1",
    "temperature": 0.2,
    "max_tokens": 1024,
    "top_k": 5,
    "enable_thinking": False,
}

SYSTEM_PROMPT = """You are the Connectome Lab Research Assistant at Seoul National University, 
led by Prof. Jiook Cha. Your role is to help external researchers and visitors understand 
the lab's research publications accurately.

Rules:
1. Answer ONLY based on the provided context from the lab's publications.
2. If the context doesn't contain relevant information, say so clearly.
3. Cite specific papers when referencing findings.
4. Use clear, accessible language while maintaining scientific accuracy.
5. For methodology questions, provide sufficient technical detail.
6. Respond in the same language as the user's question (Korean or English)."""


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
        json.dump(config, f, indent=2)


class RAGChatbot:
    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self.vectorstore = None
        self.retriever = None
        self.llm = None
        self._load_vectorstore()
        self._load_llm()

    def _load_vectorstore(self) -> None:
        if not INDEX_DIR.exists():
            logger.warning(f"FAISS index not found at {INDEX_DIR}. Run ingest.py first.")
            return

        mode = self.config["mode"]
        if mode == "nvidia_cloud":
            from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
            embedder = NVIDIAEmbeddings(
                model=self.config["nvidia_embed_model"],
                api_key=self.config["nvidia_api_key"],
            )
        else:
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

        if mode == "nvidia_cloud":
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
            self.llm = ChatNVIDIA(
                model=self.config["nvidia_model"],
                api_key=self.config["nvidia_api_key"],
                temperature=self.config["temperature"],
                max_tokens=self.config["max_tokens"],
            )
            logger.info(f"LLM loaded: NVIDIA Cloud ({self.config['nvidia_model']})")

        elif mode == "nvidia_nim":
            from openai import OpenAI
            self.llm_client = OpenAI(
                base_url=self.config["nim_endpoint"],
                api_key="not-needed",
            )
            logger.info(f"LLM loaded: NVIDIA NIM ({self.config['nim_endpoint']})")

        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            model_name = self.config["local_llm"]
            logger.info(f"Loading local LLM: {model_name} ...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )
            self.local_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=self.config["max_tokens"],
                temperature=self.config["temperature"],
                do_sample=True,
            )
            logger.info(f"LLM loaded: Local ({model_name})")

    def retrieve(self, query: str) -> list[dict]:
        if not self.retriever:
            return []
        docs = self.retriever.invoke(query)
        return [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source_file", "unknown"),
                "page": doc.metadata.get("page", "?"),
                "title": doc.metadata.get("paper_title", ""),
            }
            for doc in docs
        ]

    def generate(self, query: str, context_docs: list[dict]) -> str:
        context = "\n\n---\n\n".join(
            f"[Source: {d['title']} (p.{d['page']})]\n{d['content']}"
            for d in context_docs
        )

        user_message = f"Context from lab publications:\n{context}\n\n---\nUser question: {query}"
        mode = self.config["mode"]

        if mode == "nvidia_nim":
            response = self.llm_client.chat.completions.create(
                model=self.config["nvidia_model"],
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=self.config["temperature"],
                max_tokens=self.config["max_tokens"],
            )
            return response.choices[0].message.content

        elif mode == "nvidia_cloud":
            from langchain_core.messages import HumanMessage, SystemMessage
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_message),
            ]
            response = self.llm.invoke(messages)
            return response.content

        else:
            prompt = f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n<|user|>\n{user_message}<|end|>\n<|assistant|>\n"
            result = self.local_pipeline(prompt, return_full_text=False)
            return result[0]["generated_text"].strip()

    def chat(self, query: str) -> tuple[str, list[dict]]:
        if not query.strip():
            return "Please enter a question.", []
        context_docs = self.retrieve(query)
        if not context_docs:
            return (
                "No relevant documents found. Please make sure the FAISS index is built "
                "(run `python ingest.py` first).",
                [],
            )
        answer = self.generate(query, context_docs)
        return answer, context_docs


def create_ui(chatbot: RAGChatbot) -> gr.Blocks:
    """Build the Gradio interface."""

    with gr.Blocks(
        title="SNU Connectome Lab — Research Chatbot",
    ) as app:
        gr.HTML("""
        <div class="header">
            <h1>SNU Connectome Lab</h1>
            <p>서울대학교 뇌연결체 연구실 — AI Research Assistant</p>
            <p style="font-size:12px; color:#9ca3af;">
                Powered by NVIDIA Nemotron · Prof. Jiook Cha's Publications
            </p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=3):
                chatbox = gr.Chatbot(
                    label="Chat",
                    height=500,
                    show_label=False,
                )
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Ask about our research... (e.g., 'What methods did the lab use for Alzheimer's prediction?')",
                        show_label=False,
                        scale=5,
                        container=False,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                gr.Examples(
                    examples=[
                        "What are the main research areas of the Connectome Lab?",
                        "How did the lab use machine learning for Alzheimer's disease prediction?",
                        "What neuroimaging techniques are used in the lab's studies?",
                        "차지욱 교수님의 공포 일반화(fear generalization) 연구에 대해 설명해주세요.",
                        "해마(hippocampus)와 불안장애의 관계에 대한 연구 결과를 알려주세요.",
                    ],
                    inputs=msg_input,
                )

            with gr.Column(scale=1):
                gr.Markdown("### Retrieved Sources")
                sources_display = gr.HTML(value="<p style='color:#9ca3af;'>Sources will appear here after a query.</p>")

                with gr.Accordion("Settings", open=False):
                    mode_selector = gr.Radio(
                        choices=["local", "nvidia_cloud", "nvidia_nim"],
                        value=chatbot.config["mode"],
                        label="Inference Mode",
                    )
                    api_key_input = gr.Textbox(
                        value=chatbot.config.get("nvidia_api_key", ""),
                        label="NVIDIA API Key",
                        type="password",
                    )
                    top_k_slider = gr.Slider(1, 10, value=chatbot.config["top_k"], step=1, label="Top-K Documents")
                    temp_slider = gr.Slider(0.0, 1.0, value=chatbot.config["temperature"], step=0.05, label="Temperature")

                gr.Markdown(
                    "---\n*Built for [SNU Connectome Lab](https://connectomelab.snu.ac.kr)*\n\n"
                    f"*Index: {INDEX_DIR}*"
                )

        def respond(message: str, chat_history: list) -> tuple:
            answer, sources = chatbot.chat(message)
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": answer})

            if sources:
                html = ""
                for s in sources:
                    html += (
                        f'<div class="source-card">'
                        f'<strong>{s["title"]}</strong><br>'
                        f'<span style="color:#6b7280;">Page {s["page"]} · {s["source"]}</span>'
                        f'</div>'
                    )
            else:
                html = "<p style='color:#9ca3af;'>No sources retrieved.</p>"

            return chat_history, html, ""

        def update_settings(mode: str, api_key: str, top_k: int, temp: float) -> str:
            chatbot.config["mode"] = mode
            chatbot.config["nvidia_api_key"] = api_key
            chatbot.config["top_k"] = top_k
            chatbot.config["temperature"] = temp
            save_config(chatbot.config)
            return f"Settings updated: mode={mode}, top_k={top_k}, temp={temp}"

        send_btn.click(respond, [msg_input, chatbox], [chatbox, sources_display, msg_input])
        msg_input.submit(respond, [msg_input, chatbox], [chatbox, sources_display, msg_input])

        settings_output = gr.Textbox(visible=False)
        mode_selector.change(
            update_settings,
            [mode_selector, api_key_input, top_k_slider, temp_slider],
            settings_output,
        )

    return app


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Connectome Lab RAG Chatbot")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=7860, help="Port number")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--mode", choices=["local", "nvidia_cloud", "nvidia_nim"], default=None)
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
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css="""
        .header { text-align: center; padding: 20px 0; }
        .header h1 { color: #1e3a5f; margin-bottom: 4px; }
        .header p { color: #6b7280; font-size: 14px; }
        .source-card {
            background: #f8fafc; border-left: 3px solid #3b82f6;
            padding: 8px 12px; margin: 4px 0; border-radius: 4px; font-size: 13px;
        }
        """,
    )


if __name__ == "__main__":
    main()
