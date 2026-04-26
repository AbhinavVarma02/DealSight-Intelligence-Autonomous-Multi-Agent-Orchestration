"""Gradio web UI for DealSight Intelligence.

Renders a live dashboard of tracked deals, the best current opportunity,
and the agent log. A button or a 5-minute timer kicks off a planning
cycle on a background thread; log lines stream into the UI through a
queue while the run executes.
"""

import argparse
import html
import inspect
import logging
import queue
import threading
import time

from dealsight_intelligence.app.deal_agent_framework import DealAgentFramework
from dealsight_intelligence.app.log_utils import reformat

MODAL_HTML = """
<div id="pir-modal" data-hidden style="position:fixed; top:0; left:0; right:0; bottom:0; background:rgba(0,0,0,0.6); z-index:1000; align-items:center; justify-content:center;">
  <div style="background:var(--pir-panel); border-radius:12px; padding:24px; max-width:520px; width:90%; max-height:80vh; overflow-y:auto; box-shadow:var(--pir-shadow);">
    <h2 id="pir-modal-title" style="margin:0 0 14px; color:var(--pir-ink); font-size:1.28rem; font-weight:800;"></h2>
    <div id="pir-modal-body" style="margin:0; color:var(--pir-muted); line-height:1.7; font-size:0.95rem; white-space:pre-wrap; word-break:break-word;"></div>
    <button id="pir-modal-close" style="margin-top:18px; background:var(--pir-green); color:#fff; border:none; padding:10px 18px; border-radius:6px; cursor:pointer; font-weight:700; font-size:0.95rem; transition:filter 0.15s;">Close</button>
  </div>
</div>
<style>
  #pir-modal { display: flex !important; }
  #pir-modal[data-hidden] { display: none !important; }
  #pir-modal-close:hover { filter: brightness(1.1); }
</style>
<script>
  window.showDealModal = function(title, description) {
    const modal = document.getElementById('pir-modal');
    document.getElementById('pir-modal-title').textContent = title;
    document.getElementById('pir-modal-body').textContent = description;
    modal.removeAttribute('data-hidden');
  };
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initModal);
  } else {
    initModal();
  }
  function initModal() {
    const closeBtn = document.getElementById('pir-modal-close');
    const modal = document.getElementById('pir-modal');
    if (closeBtn) closeBtn.addEventListener('click', () => modal.setAttribute('data-hidden', ''));
    if (modal) modal.addEventListener('click', (e) => { if (e.target === modal) modal.setAttribute('data-hidden', ''); });
  }
</script>
"""

APP_CSS = """
:root, .light {
    --pir-ink: #1a1d1b;
    --pir-muted: #5e6761;
    --pir-paper: #f6f8f7;
    --pir-panel: #ffffff;
    --pir-panel-soft: #f3f7f5;
    --pir-line: #dce3df;
    --pir-line-soft: #ecf0ed;
    --pir-green: #008a6e;
    --pir-green-accent: #7de2c6;
    --pir-red: #e84a5f;
    --pir-gold: #f2b705;
    --pir-hero-bg: linear-gradient(135deg, #1a1d1b 0%, #2a3d36 100%);
    --pir-hero-fg: #ffffff;
    --pir-hero-copy: #e7eee9;
    --pir-log-bg: #fffdf8;
    --pir-log-border: #f0b47a;
    --pir-log-line: #ebe5d8;
    --pir-shadow: 0 1px 3px rgba(0, 0, 0, 0.04), 0 4px 14px rgba(0, 0, 0, 0.05);
}

.dark {
    --pir-ink: #ecf0ed;
    --pir-muted: #97a39c;
    --pir-paper: #0f1311;
    --pir-panel: #1a1f1c;
    --pir-panel-soft: #232a26;
    --pir-line: #2c332f;
    --pir-line-soft: #232a26;
    --pir-green: #2dd4a8;
    --pir-green-accent: #7de2c6;
    --pir-red: #ff7a8c;
    --pir-gold: #ffcf3f;
    --pir-hero-bg: linear-gradient(135deg, #0d1f18 0%, #163d31 100%);
    --pir-hero-fg: #ffffff;
    --pir-hero-copy: #cfded6;
    --pir-log-bg: #1a1f1c;
    --pir-log-border: #3a2f1f;
    --pir-log-line: #2c332f;
    --pir-shadow: 0 1px 3px rgba(0, 0, 0, 0.4), 0 4px 14px rgba(0, 0, 0, 0.35);
}

.gradio-container {
    background: var(--pir-paper) !important;
    color: var(--pir-ink);
}

#pir-shell {
    max-width: 1180px;
    margin: 0 auto;
    padding: 20px 18px 30px;
}

#pir-shell, #pir-shell * {
    letter-spacing: 0;
}

.pir-hero {
    background: var(--pir-hero-bg);
    color: var(--pir-hero-fg);
    border: 1px solid transparent;
    border-radius: 12px;
    padding: 28px;
    display: grid;
    grid-template-columns: minmax(0, 1.1fr) minmax(280px, 0.9fr);
    gap: 22px;
    align-items: stretch;
    box-shadow: var(--pir-shadow);
}

.pir-kicker {
    color: var(--pir-green-accent);
    font-size: 0.86rem;
    font-weight: 800;
    margin: 0 0 10px;
    text-transform: uppercase;
    letter-spacing: 0.05em !important;
}

.pir-hero h1 {
    color: var(--pir-hero-fg) !important;
    font-size: 2.45rem;
    line-height: 1.03;
    margin: 0;
    max-width: 12ch;
}

.pir-hero-copy {
    color: var(--pir-hero-copy);
    font-size: 1.02rem;
    line-height: 1.55;
    margin: 14px 0 0;
    max-width: 56ch;
}

.pir-best {
    background: var(--pir-panel);
    color: var(--pir-ink);
    border: 1px solid var(--pir-line-soft);
    border-radius: 10px;
    padding: 18px;
    display: grid;
    gap: 12px;
}

.pir-best-label {
    color: var(--pir-muted);
    font-size: 0.86rem;
    font-weight: 800;
    margin: 0;
    text-transform: uppercase;
}

.pir-best-title {
    color: var(--pir-ink);
    font-size: 1.18rem;
    font-weight: 800;
    line-height: 1.28;
    margin: 0;
}

.pir-price-row {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 8px;
}

.pir-price-chip {
    background: var(--pir-panel-soft);
    border: 1px solid var(--pir-line);
    border-radius: 8px;
    padding: 10px;
    min-height: 68px;
}

.pir-price-chip span {
    color: var(--pir-muted);
    display: block;
    font-size: 0.78rem;
    font-weight: 800;
    margin-bottom: 6px;
    text-transform: uppercase;
}

.pir-price-chip strong {
    color: var(--pir-ink);
    display: block;
    font-size: 1.02rem;
}

.pir-price-chip.savings strong {
    color: var(--pir-green);
}

.pir-best-link {
    color: #ffffff !important;
    background: var(--pir-green);
    border-radius: 8px;
    display: inline-flex;
    font-weight: 800;
    justify-content: center;
    padding: 10px 12px;
    text-decoration: none !important;
    transition: transform 0.15s ease, filter 0.15s ease;
}

.pir-best-link:hover {
    filter: brightness(1.08);
    transform: translateY(-1px);
}

.pir-scoreboard {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
    margin: 14px 0;
}

.pir-stat {
    background: var(--pir-panel);
    border: 1px solid var(--pir-line);
    border-left: 5px solid var(--pir-green);
    border-radius: 10px;
    padding: 14px;
    min-height: 96px;
    box-shadow: var(--pir-shadow);
}

.pir-stat:nth-child(2) { border-left-color: var(--pir-red); }
.pir-stat:nth-child(3) { border-left-color: var(--pir-gold); }
.pir-stat:nth-child(4) { border-left-color: var(--pir-muted); }

.pir-stat span {
    color: var(--pir-muted);
    display: block;
    font-size: 0.82rem;
    font-weight: 800;
    margin-bottom: 8px;
    text-transform: uppercase;
}

.pir-stat strong {
    color: var(--pir-ink);
    display: block;
    font-size: 1.5rem;
    line-height: 1.1;
}

.pir-toolbar {
    align-items: center;
    margin: 8px 0 12px;
}

#run-scan {
    border-radius: 8px;
    min-height: 44px;
    background: var(--pir-green) !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 700 !important;
}

#run-scan:hover {
    filter: brightness(1.08);
}

.pir-note {
    color: var(--pir-muted);
    font-size: 0.95rem;
    line-height: 1.45;
    padding: 8px 0;
}

.pir-section-title {
    color: var(--pir-ink);
    font-size: 1.12rem;
    font-weight: 900;
    margin: 4px 0 10px;
}

#opportunities {
    border: 1px solid var(--pir-line);
    border-radius: 10px;
    overflow: hidden;
    background: var(--pir-panel);
    box-shadow: var(--pir-shadow);
}

#opportunities .table-wrap,
#opportunities .svelte-virtual-table-viewport,
#opportunities [class*="table"] {
    background: var(--pir-panel) !important;
}

#opportunities .cell-wrap,
#opportunities [class*="cell"] {
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    white-space: nowrap !important;
    max-width: 100% !important;
}

#opportunities .cell-wrap span,
#opportunities [class*="cell"] span {
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    white-space: nowrap !important;
    display: block !important;
    max-width: 100% !important;
}

#opportunities table {
    font-size: 0.9rem;
    background: var(--pir-panel) !important;
    color: var(--pir-ink) !important;
    width: 100%;
    table-layout: fixed;
    border-collapse: collapse;
}

#opportunities col:nth-child(1) { width: 44%; }
#opportunities col:nth-child(2),
#opportunities col:nth-child(3),
#opportunities col:nth-child(4) { width: 12%; }
#opportunities col:nth-child(5) { width: 20%; }

#opportunities th,
#opportunities td {
    padding: 12px 14px !important;
    line-height: 1.4 !important;
    vertical-align: middle;
    border-bottom: 1px solid var(--pir-line-soft) !important;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

#opportunities th {
    background: var(--pir-panel-soft) !important;
    color: var(--pir-ink) !important;
    font-weight: 700 !important;
    text-align: left !important;
    border-bottom: 2px solid var(--pir-line) !important;
    font-size: 0.82rem !important;
    text-transform: uppercase;
    letter-spacing: 0.04em !important;
}

#opportunities td {
    background: var(--pir-panel) !important;
    color: var(--pir-ink) !important;
}

#opportunities td:first-child {
    cursor: pointer;
    font-weight: 500;
}

#opportunities tr:hover td {
    background: var(--pir-panel-soft) !important;
}

#opportunities tr:hover td:first-child {
    color: var(--pir-green) !important;
}

#opportunities td:nth-child(2),
#opportunities td:nth-child(3) {
    font-variant-numeric: tabular-nums;
    text-align: right;
}

#opportunities th:nth-child(2),
#opportunities th:nth-child(3),
#opportunities th:nth-child(4) {
    text-align: right !important;
}

#opportunities td:nth-child(4) {
    color: var(--pir-green) !important;
    font-weight: 800;
    font-variant-numeric: tabular-nums;
    text-align: right;
}

#opportunities td:nth-child(5) {
    color: var(--pir-muted) !important;
    font-size: 0.82rem;
    font-family: Consolas, Monaco, "Courier New", monospace;
}

.pir-log-panel {
    background: var(--pir-log-bg);
    border: 1px solid var(--pir-log-border);
    border-radius: 10px;
    color: var(--pir-ink);
    height: 360px;
    overflow-y: auto;
    padding: 14px;
    box-shadow: var(--pir-shadow);
}

.pir-log-heading {
    color: var(--pir-green);
    font-size: 0.86rem;
    font-weight: 900;
    margin-bottom: 10px;
    text-transform: uppercase;
}

.pir-log-line {
    border-bottom: 1px solid var(--pir-log-line);
    color: var(--pir-ink);
    font-family: Consolas, Monaco, "Courier New", monospace;
    font-size: 0.88rem;
    line-height: 1.55;
    padding: 6px 0;
    white-space: normal;
    overflow-wrap: anywhere;
}

.pir-log-line:last-child {
    border-bottom: 0;
}

.pir-empty {
    color: var(--pir-hero-copy);
    background: rgba(255, 255, 255, 0.06);
    border: 1px dashed rgba(255, 255, 255, 0.25);
    border-radius: 8px;
    padding: 14px;
}

@media (max-width: 860px) {
    .pir-hero,
    .pir-scoreboard {
        grid-template-columns: 1fr;
    }

    .pir-hero h1 {
        font-size: 1.9rem;
        max-width: 100%;
    }

    .pir-price-row {
        grid-template-columns: 1fr;
    }
}
"""

TABLE_HEADERS = ["Deal", "Price", "Estimate", "Savings", "URL"]


class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))


def money(value: float) -> str:
    return f"${value:,.2f}"


def truncate(text: str, limit: int = 55) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def short_url(url: str, limit: int = 32) -> str:
    cleaned = url.replace("https://", "").replace("http://", "").rstrip("/")
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1] + "…"


def table_for(opportunities):
    return [
        [
            truncate(item.deal.product_description),
            money(item.deal.price),
            money(item.estimate),
            money(item.discount),
            short_url(item.deal.url),
        ]
        for item in opportunities
    ]


def dashboard_for(opportunities):
    if not opportunities:
        return """
        <section class="pir-hero">
            <div>
                <p class="pir-kicker">Live deal desk</p>
                <h1>Beat the sticker.</h1>
                <p class="pir-hero-copy">Fresh scans price each find against the model ensemble, then keep the best opportunities ready for review.</p>
            </div>
            <div class="pir-empty">Waiting for the next scan.</div>
        </section>
        <section class="pir-scoreboard">
            <div class="pir-stat"><span>Tracked deals</span><strong>0</strong></div>
            <div class="pir-stat"><span>Top savings</span><strong>$0.00</strong></div>
            <div class="pir-stat"><span>Average savings</span><strong>$0.00</strong></div>
            <div class="pir-stat"><span>Refresh</span><strong>5 min</strong></div>
        </section>
        """

    top = max(opportunities, key=lambda item: item.discount)
    positive_discounts = [max(0, item.discount) for item in opportunities]
    average_discount = sum(positive_discounts) / len(positive_discounts)
    deal_count = len(opportunities)
    title = html.escape(top.deal.product_description)
    url = html.escape(top.deal.url, quote=True)
    return f"""
    <section class="pir-hero">
        <div>
            <p class="pir-kicker">Live deal desk</p>
            <h1>Beat the sticker.</h1>
            <p class="pir-hero-copy">Fresh scans price each find against the model ensemble, then keep the best opportunities ready for review.</p>
        </div>
        <article class="pir-best">
            <p class="pir-best-label">Best find right now</p>
            <h2 class="pir-best-title">{title}</h2>
            <div class="pir-price-row">
                <div class="pir-price-chip"><span>Price</span><strong>{money(top.deal.price)}</strong></div>
                <div class="pir-price-chip"><span>Estimate</span><strong>{money(top.estimate)}</strong></div>
                <div class="pir-price-chip savings"><span>Savings</span><strong>{money(top.discount)}</strong></div>
            </div>
            <a class="pir-best-link" href="{url}" target="_blank" rel="noreferrer">Open deal</a>
        </article>
    </section>
    <section class="pir-scoreboard">
        <div class="pir-stat"><span>Tracked deals</span><strong>{deal_count}</strong></div>
        <div class="pir-stat"><span>Top savings</span><strong>{money(top.discount)}</strong></div>
        <div class="pir-stat"><span>Average savings</span><strong>{money(average_discount)}</strong></div>
        <div class="pir-stat"><span>Refresh</span><strong>5 min</strong></div>
    </section>
    """


def html_for(log_data):
    if log_data:
        output = "".join(f'<div class="pir-log-line">{line}</div>' for line in log_data[-24:])
    else:
        output = '<div class="pir-log-line">Waiting for agent output.</div>'
    return (
        '<div class="pir-log-panel">'
        '<div class="pir-log-heading">Agent log</div>'
        f"{output}</div>"
    )


def setup_logging(log_queue):
    handler = QueueHandler(log_queue)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S %z"))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)
    return handler


def teardown_logging(handler):
    logging.getLogger().removeHandler(handler)
    handler.close()


class App:
    def __init__(self):
        self.agent_framework = None
        self._framework_lock = threading.Lock()

    def get_agent_framework(self):
        if self.agent_framework:
            return self.agent_framework
        with self._framework_lock:
            if not self.agent_framework:
                self.agent_framework = DealAgentFramework()
                self.agent_framework.init_agents_as_needed()
        return self.agent_framework

    def run_once(self):
        return self.get_agent_framework().run()

    def run(self):
        try:
            import gradio as gr
        except ImportError as exc:
            raise RuntimeError('Install app dependencies with: python -m pip install -e ".[app]"') from exc

        try:
            theme = gr.themes.Soft(
                primary_hue="emerald",
                secondary_hue="teal",
                neutral_hue="slate",
                font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
            )
        except Exception:
            theme = None

        blocks_kwargs = {"title": "DealSight Intelligence", "fill_width": True}
        if theme is not None:
            blocks_kwargs["theme"] = theme
        launch_kwargs = {"share": False, "inbrowser": False}
        if "css" in inspect.signature(gr.Blocks).parameters:
            blocks_kwargs["css"] = APP_CSS
        elif "css" in inspect.signature(gr.Blocks.launch).parameters:
            launch_kwargs["css"] = APP_CSS

        with gr.Blocks(**blocks_kwargs) as ui:
            log_data = gr.State([])

            def update_output(current_logs, log_queue, result_queue):
                initial_memory = self.get_agent_framework().memory
                initial_result = table_for(initial_memory)
                current_dashboard = dashboard_for(initial_memory)
                final_result = None
                while True:
                    try:
                        message = log_queue.get_nowait()
                        current_logs.append(reformat(message))
                        yield current_logs, current_dashboard, html_for(current_logs), final_result or initial_result
                    except queue.Empty:
                        try:
                            status, payload = result_queue.get_nowait()
                            if status == "error":
                                escaped = html.escape(str(payload))
                                current_logs.append(f'<span style="color:#ff8a8a">Run failed: {escaped}</span>')
                                final_result = initial_result
                                current_dashboard = dashboard_for(initial_memory)
                            else:
                                final_result = payload
                                current_dashboard = dashboard_for(self.get_agent_framework().memory)
                            yield current_logs, current_dashboard, html_for(current_logs), final_result
                            break
                        except queue.Empty:
                            time.sleep(0.1)

            def do_run():
                return table_for(self.run_once())

            def run_with_logging(current_logs):
                log_queue = queue.Queue()
                result_queue = queue.Queue()
                handler = setup_logging(log_queue)

                def worker():
                    try:
                        result_queue.put(("ok", do_run()))
                    except Exception as exc:
                        result_queue.put(("error", exc))

                thread = threading.Thread(target=worker, daemon=True)
                thread.start()
                try:
                    yield from update_output(current_logs, log_queue, result_queue)
                finally:
                    teardown_logging(handler)

            def do_select(selected_index: gr.SelectData):
                row = selected_index.index[0]
                framework = self.get_agent_framework()
                if row < len(framework.memory):
                    item = framework.memory[row]
                    title = html.escape(item.deal.product_description)
                    desc = f"Price: {money(item.deal.price)}\nEstimate: {money(item.estimate)}\nSavings: {money(item.discount)}\n\nURL: {item.deal.url}"
                    desc = html.escape(desc)
                    return gr.update(value=f'<script>window.showDealModal({repr(title)}, {repr(desc)})</script>')

            with gr.Column(elem_id="pir-shell"):
                dashboard = gr.HTML(dashboard_for([]))
                with gr.Row(elem_classes=["pir-toolbar"]):
                    run_button = gr.Button("Run scan now", variant="primary", elem_id="run-scan")
                    gr.HTML('<div class="pir-note">Auto-scan runs every 5 minutes. Click a deal to view full details.</div>')
                gr.HTML('<div class="pir-section-title">Opportunities</div>')
                opportunities_dataframe = gr.Dataframe(
                    headers=TABLE_HEADERS,
                    datatype=["str", "str", "str", "str", "str"],
                    wrap=False,
                    row_count=(10, "fixed"),
                    column_count=(5, "fixed"),
                    max_height=430,
                    interactive=False,
                    elem_id="opportunities",
                )
                gr.HTML(MODAL_HTML)
                modal_trigger = gr.HTML(value="", visible=False)
                gr.HTML('<div class="pir-section-title">Run Log</div>')
                logs = gr.HTML(html_for([]))
            ui.load(run_with_logging, inputs=[log_data], outputs=[log_data, dashboard, logs, opportunities_dataframe])
            run_button.click(run_with_logging, inputs=[log_data], outputs=[log_data, dashboard, logs, opportunities_dataframe])
            timer = gr.Timer(value=300, active=True)
            timer.tick(run_with_logging, inputs=[log_data], outputs=[log_data, dashboard, logs, opportunities_dataframe])
            opportunities_dataframe.select(do_select, outputs=[modal_trigger])
        ui.launch(**launch_kwargs)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run one planning cycle and print the memory table")
    args = parser.parse_args(argv)
    app = App()
    if args.once:
        for row in table_for(app.run_once()):
            print(row)
        return
    app.run()


if __name__ == "__main__":
    main()
