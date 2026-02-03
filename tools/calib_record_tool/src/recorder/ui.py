from __future__ import annotations

import threading
import queue
from dataclasses import asdict
from pathlib import Path
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

from .config import RecorderConfig
from . import run_pipeline


class RecorderUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Calib Record Tool - Recorder UI (Minimal)")
        self.geometry("900x600")

        self.log_q: "queue.Queue[str]" = queue.Queue()
        self._worker: threading.Thread | None = None

        self._build()

        # periodic log pump
        self.after(50, self._pump_logs)

    def _build(self):
        frm = ttk.Frame(self, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)

        # Inputs
        grid = ttk.Frame(frm)
        grid.pack(fill=tk.X)

        self.var_device = tk.StringVar(value="3D USB Camera")
        self.var_size = tk.StringVar(value="3200x1200")
        self.var_fps = tk.StringVar(value="60")
        self.var_seconds = tk.StringVar(value="120")
        self.var_outdir = tk.StringVar(value=".")
        self.var_outroot = tk.StringVar(value="")

        def row(r, label, var):
            ttk.Label(grid, text=label, width=12).grid(row=r, column=0, sticky="w", padx=4, pady=4)
            ttk.Entry(grid, textvariable=var).grid(row=r, column=1, sticky="we", padx=4, pady=4)

        grid.columnconfigure(1, weight=1)

        row(0, "device", self.var_device)
        row(1, "size", self.var_size)
        row(2, "fps", self.var_fps)
        row(3, "seconds", self.var_seconds)
        row(4, "outdir", self.var_outdir)
        row(5, "outroot", self.var_outroot)

        # Buttons
        btns = ttk.Frame(frm)
        btns.pack(fill=tk.X, pady=(10, 0))

        self.btn_start = ttk.Button(btns, text="Start", command=self._on_start)
        self.btn_start.pack(side=tk.LEFT)

        self.btn_clear = ttk.Button(btns, text="Clear Log", command=self._clear_log)
        self.btn_clear.pack(side=tk.LEFT, padx=8)

        # Log area
        self.log = scrolledtext.ScrolledText(frm, height=25)
        self.log.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        self._append("UI ready.\n")

    def _append(self, text: str):
        self.log.insert(tk.END, text)
        self.log.see(tk.END)

    def _clear_log(self):
        self.log.delete("1.0", tk.END)

    def _pump_logs(self):
        while True:
            try:
                msg = self.log_q.get_nowait()
            except queue.Empty:
                break
            self._append(msg + "\n")
        self.after(50, self._pump_logs)

    def _on_start(self):
        if self._worker and self._worker.is_alive():
            messagebox.showwarning("Running", "Pipeline is already running.")
            return

        # Parse config
        try:
            cfg = RecorderConfig(
                device=self.var_device.get().strip(),
                size=self.var_size.get().strip(),
                fps=int(self.var_fps.get().strip()),
                seconds=int(self.var_seconds.get().strip()),
                outdir=self.var_outdir.get().strip(),
                outroot=self.var_outroot.get().strip(),
            )
        except Exception as e:
            messagebox.showerror("Invalid input", str(e))
            return

        self.btn_start.config(state=tk.DISABLED)
        self.log_q.put(f"[CONFIG] {asdict(cfg)}")

        def on_event(ev: dict):
            stage = ev.get("stage", "")
            msg = ev.get("message", "")
            self.log_q.put(f"[{stage}] {msg}")

        def worker():
            try:
                res = run_pipeline(cfg, on_event=on_event)
                self.log_q.put(f"[RESULT] returncode={res.returncode}")
                self.log_q.put(f"[RESULT] session_hint={res.session_hint}")
            except Exception as e:
                self.log_q.put(f"[EXCEPTION] {e}")
            finally:
                # Re-enable start
                self.btn_start.config(state=tk.NORMAL)

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()


def main():
    app = RecorderUI()
    app.mainloop()


if __name__ == "__main__":
    main()
