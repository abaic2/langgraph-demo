from streamlit.testing.v1 import AppTest

at = AppTest.from_file("streamlit_app.py", default_timeout=60)
at.run()
assert not at.exception, at.exception
at.chat_input[0].set_value("分析低空经济赛道2026年的投资机会").run()
assert not at.exception, at.exception
texts = "\n".join(m.value for m in at.markdown)
for tag in ("supervisor", "researcher", "analyst", "writer", "FINISH", "最终答案"):
    assert tag in texts, f"missing {tag}"
print("UI SMOKE OK")
