"""Lokal LLM via Ollama / Local LLM through Ollama (http://localhost:11434). Ingen skytjenester / no cloud.

  python llm.py "Write a clean Next.js API route that returns JSON"     # spør modellen / ask the model
  python llm.py --models                    # hva er installert, hva passer for denne maskinen / what fits this machine
  python llm.py --model qwen2.5-coder:7b "..."                          # velg modell selv / pick a model yourself
  python llm.py --selftest                  # tester klienten mot en lokal stub, uten Ollama / tests the client with a stub

Modell / Model:  --model, ellers miljøvariabelen KMBRAIN_MODEL, ellers automatisk valg (se PREFERRED).
Adresse / Host:  miljøvariabelen OLLAMA_HOST (standard http://localhost:11434).
Bare standardbiblioteket brukes, ingen ekstra pakker. / Standard library only, no extra packages.
"""

import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Callable, Dict, List, Optional, Tuple

DEFAULT_HOST = "http://localhost:11434"

# Rekkefølgen du ba om (sterkest først). De to siste er mindre reservevalg for svake maskiner.
# Your requested order, strongest first. The last two are smaller fallbacks for weak machines.
PREFERRED = ["qwen2.5-coder:14b", "qwen2.5-coder:7b", "deepseek-coder-v2", "llama3.1"]
SMALL_FALLBACKS = ["qwen2.5-coder:3b", "qwen2.5-coder:1.5b"]

# Omtrentlig nedlastingsstørrelse (GB) for standardvarianten (4-bit). Cirka-tall, ikke garantier.
# Approximate download size (GB) of the default 4-bit variant. Rough numbers, not guarantees.
SIZE_GB = {"qwen2.5-coder:14b": 9.0, "qwen2.5-coder:7b": 4.7, "deepseek-coder-v2": 8.9, "llama3.1": 4.9,
           "qwen2.5-coder:3b": 1.9, "qwen2.5-coder:1.5b": 1.0}


class OllamaError(RuntimeError):
    pass


class OllamaNotRunning(OllamaError):
    pass


def host() -> str:
    h = os.environ.get("OLLAMA_HOST", DEFAULT_HOST).strip().rstrip("/")
    return h if h.startswith("http") else f"http://{h}"


NOT_RUNNING = ("Ollama kjører ikke på {h}. Start Ollama-appen (eller kjør 'ollama serve' i en terminal). "
               "Ikke installert? Last ned fra https://ollama.com/download\n"
               "    [EN] Ollama is not running on {h}. Start the Ollama app (or run 'ollama serve' in a terminal). "
               "Not installed? Get it from https://ollama.com/download")


# ------------------------------------------------------------------ HTTP

def _open(path: str, payload: Optional[dict] = None, timeout: float = 5.0):
    url = host() + path
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        raise OllamaError(f"Ollama svarte {e.code} / Ollama answered {e.code}: {body[:200]}") from e
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
        raise OllamaNotRunning(NOT_RUNNING.format(h=host())) from e


def is_running(timeout: float = 0.7) -> bool:
    try:
        _open("/api/tags", timeout=timeout).close()
        return True
    except OllamaError:
        return False


def installed_models() -> List[str]:
    with _open("/api/tags") as r:
        return [m["name"] for m in json.loads(r.read().decode("utf-8")).get("models", [])]


# ------------------------------------------------------------------ maskin og valg / machine and choice

def hardware() -> dict:
    """RAM (totalt/ledig) og NVIDIA-VRAM i GB, så godt det lar seg gjøre. / RAM and NVIDIA VRAM, best effort."""
    ram = free = 0.0
    try:
        if os.name == "nt":
            class MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MemStatus()
            m.dwLength = ctypes.sizeof(m)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            ram, free = m.ullTotalPhys / 2**30, m.ullAvailPhys / 2**30
        else:
            ram = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30
            free = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30
    except Exception:
        pass
    vram, gpu = 0.0, ""
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
            gpu, mib = [x.strip() for x in out.rsplit(",", 1)]
            vram = float(mib) / 1024
        except Exception:
            pass
    return {"ram_gb": ram, "free_gb": free, "vram_gb": vram, "gpu": gpu}


def key_of(model: str) -> Optional[str]:
    """Hvilken kjent modell er dette? 'qwen2.5-coder:7b-instruct-q4' -> 'qwen2.5-coder:7b'."""
    m = model.lower()
    for k in PREFERRED + SMALL_FALLBACKS:
        if m == k or m.startswith(k + "-") or (":" not in k and m.startswith(k + ":")) or m.startswith(k + ":"):
            return k
    return None


def fit(model: str, hw: dict) -> str:
    """'ok' (god plass), 'tight' (går, men trangt/tregt) eller 'no' (for stor). Grov tommelfingerregel:
    modellen må ligge i RAM (eller i VRAM hvis den får plass der), og Windows trenger sitt.
    Rule of thumb: the model must fit in RAM (or VRAM if it fits there) and the OS needs room too."""
    size = SIZE_GB.get(key_of(model) or "", 0)
    if not size or not hw["ram_gb"]:
        return "ok"
    if hw["vram_gb"] and size * 1.2 <= hw["vram_gb"]:
        return "ok"  # kjører på GPU / runs on the GPU
    if size * 2 <= hw["ram_gb"]:
        return "ok"
    if size * 1.5 <= hw["ram_gb"]:
        return "tight"
    return "no"


def recommend_pull(hw: dict) -> Tuple[str, str]:
    """Sterkeste modell som ikke er 'no' på denne maskinen. (navn, fit)"""
    for m in PREFERRED + SMALL_FALLBACKS:
        f = fit(m, hw)
        if f != "no":
            return m, f
    return SMALL_FALLBACKS[-1], "tight"


def pick_model(installed: List[str], explicit: Optional[str] = None, hw: Optional[dict] = None) -> Tuple[str, Optional[str]]:
    """Velg modell: --model / KMBRAIN_MODEL, ellers første installerte i PREFERRED som passer maskinen.
    Returnerer (modellnavn, advarsel|None). Kaster OllamaError med råd hvis ingenting er installert."""
    hw = hw or hardware()
    explicit = explicit or os.environ.get("KMBRAIN_MODEL")
    if explicit:
        for name in installed:
            if name == explicit or name.startswith(explicit + ":") or name.startswith(explicit + "-"):
                return name, None
        raise OllamaError(f"Modellen '{explicit}' er ikke lastet ned. Kjør: ollama pull {explicit}\n"
                          f"    [EN] Model '{explicit}' is not pulled. Run: ollama pull {explicit}")
    first_too_big = None
    for wanted in PREFERRED + SMALL_FALLBACKS:
        for name in installed:
            if key_of(name) == wanted:
                f = fit(name, hw)
                if f == "no":
                    first_too_big = first_too_big or name
                    continue
                return name, ("Trang plass: dette blir tregt. / Tight fit: expect it to be slow." if f == "tight" else None)
    if first_too_big:
        return first_too_big, (f"{first_too_big} er sannsynligvis for stor for denne maskinen og kan bli veldig treg. / "
                               f"{first_too_big} is probably too big for this machine and may be very slow.")
    best, f = recommend_pull(hw)
    raise OllamaError(f"Ingen anbefalt modell er lastet ned. Anbefalt for denne maskinen: ollama pull {best}\n"
                      f"    [EN] No recommended model is pulled yet. Recommended for this machine: ollama pull {best}")


# ------------------------------------------------------------------ chat

def chat(messages: List[dict], model: str, on_token: Optional[Callable[[str], None]] = None,
         temperature: float = 0.2, num_ctx: int = 4096, timeout: float = 900.0) -> str:
    """Send en samtale og få svaret (strømmes til on_token hvis gitt). Første kall kan ta lang tid mens
    modellen lastes. / Send a conversation; the first call can be slow while the model loads."""
    payload = {"model": model, "messages": messages, "stream": True,
               "options": {"temperature": temperature, "num_ctx": num_ctx}}
    try:
        resp = _open("/api/chat", payload, timeout=timeout)
    except OllamaError as e:
        if "404" in str(e):
            raise OllamaError(f"Modellen '{model}' finnes ikke lokalt. Kjør: ollama pull {model}\n"
                              f"    [EN] Model '{model}' is not available locally. Run: ollama pull {model}") from e
        raise
    parts: List[str] = []
    with resp:
        for raw in resp:
            if not raw.strip():
                continue
            obj = json.loads(raw.decode("utf-8"))
            if obj.get("error"):
                raise OllamaError(str(obj["error"]))
            tok = obj.get("message", {}).get("content", "")
            if tok:
                parts.append(tok)
                if on_token:
                    on_token(tok)
            if obj.get("done"):
                break
    return "".join(parts)


def generate(prompt: str, system: Optional[str] = None, model: Optional[str] = None,
             on_token: Optional[Callable[[str], None]] = None, **kw) -> Tuple[str, str]:
    """Enkel inngang: velger modell selv. Returnerer (svar, modellnavn). / Simple entry point."""
    name, _ = pick_model(installed_models(), model)
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    return chat(msgs, name, on_token, **kw), name


# ------------------------------------------------------------------ CLI

def show_models() -> int:
    hw = hardware()
    gpu = f", GPU {hw['gpu']} {hw['vram_gb']:.0f} GB VRAM" if hw["gpu"] else ", ingen NVIDIA-GPU funnet / no NVIDIA GPU found"
    print(f"Maskin / Machine: {hw['ram_gb']:.1f} GB RAM ({hw['free_gb']:.1f} GB ledig/free){gpu}")
    best, f = recommend_pull(hw)
    try:
        have = installed_models()
        running = True
    except OllamaNotRunning:
        have, running = [], False
        print(NOT_RUNNING.format(h=host()))
    if running:
        print(f"Installert / Installed: {', '.join(have) or '(ingen/none)'}")
    print("\nAnbefalt rekkefølge / Preferred order (fit = passer denne maskinen / fits this machine):")
    for m in PREFERRED + SMALL_FALLBACKS:
        mark = "installert/installed" if any(key_of(n) == m for n in have) else ""
        print(f"  {m:<22} ~{SIZE_GB[m]:>4.1f} GB   fit: {fit(m, hw):<6} {mark}")
    note = " (trangt, men mulig / tight but possible)" if f == "tight" else ""
    print(f"\nSterkeste som passer / Strongest that fits: ollama pull {best}{note}")
    alt = next((m for m in SMALL_FALLBACKS if fit(m, hw) == "ok"), None)
    if f == "tight" and alt:
        print(f"Raskere og tryggere / Faster and safer: ollama pull {alt}")
    if running:
        try:
            name, warn = pick_model(have, None, hw)
            print(f"\nBrukes nå / In use now: {name}" + (f"\n  ! {warn}" if warn else ""))
        except OllamaError as e:
            print(f"\n{e}")
    return 0


def selftest() -> int:
    """Starter en falsk Ollama-server lokalt og tester klienten uten ekte Ollama. Tester IKKE kvaliteten på en ekte modell.
    Spins up a fake Ollama and tests the client without real Ollama. Does NOT test real model quality."""
    import http.server
    import threading

    class Fake(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            body = json.dumps({"models": [{"name": "llama3.1:8b"}, {"name": "qwen2.5-coder:7b-instruct-q4_K_M"}]}).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if req["model"] == "missing:1b":
                self.send_response(404); self.end_headers(); self.wfile.write(b'{"error":"model not found"}'); return
            self.send_response(200); self.end_headers()
            for tok in ("Hel", "lo ", "from ", req["model"]):
                self.wfile.write((json.dumps({"message": {"content": tok}, "done": False}) + "\n").encode())
            self.wfile.write((json.dumps({"message": {"content": ""}, "done": True}) + "\n").encode())

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    old = os.environ.get("OLLAMA_HOST")
    os.environ["OLLAMA_HOST"] = f"127.0.0.1:{srv.server_address[1]}"
    ok = True

    def check(name: str, cond: bool):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'OK ' if cond else 'FEIL/FAIL'} {name}")

    try:
        big = {"ram_gb": 32.0, "free_gb": 20.0, "vram_gb": 0.0, "gpu": ""}
        weak = {"ram_gb": 7.9, "free_gb": 3.0, "vram_gb": 2.0, "gpu": "GTX 1050"}
        have = installed_models()
        check("list models", have == ["llama3.1:8b", "qwen2.5-coder:7b-instruct-q4_K_M"])
        check("prefers qwen coder over llama (big machine)", pick_model(have, None, big)[0].startswith("qwen2.5-coder:7b"))
        check("explicit model by prefix", pick_model(have, "llama3.1", big)[0] == "llama3.1:8b")
        check("weak machine recommends 7b as tight", recommend_pull(weak) == ("qwen2.5-coder:7b", "tight"))
        check("14b not recommended on 8 GB RAM", fit("qwen2.5-coder:14b", weak) == "no")
        check("14b fits 32 GB RAM", fit("qwen2.5-coder:14b", big) == "ok")
        try:
            pick_model(have, "deepseek-coder-v2", big)
            check("missing explicit model raises", False)
        except OllamaError as e:
            check("missing explicit model raises with pull hint", "ollama pull deepseek-coder-v2" in str(e))
        toks: List[str] = []
        out = chat([{"role": "user", "content": "hi"}], "llama3.1:8b", toks.append)
        check("streaming chat joins tokens", out == "Hello from llama3.1:8b" and len(toks) == 4)
        try:
            chat([{"role": "user", "content": "hi"}], "missing:1b")
            check("404 -> pull hint", False)
        except OllamaError as e:
            check("404 -> pull hint", "ollama pull missing:1b" in str(e))
        srv.shutdown()
        os.environ["OLLAMA_HOST"] = "127.0.0.1:1"  # ingen server her / nothing listens here
        try:
            installed_models()
            check("not running -> clear error", False)
        except OllamaNotRunning as e:
            check("not running -> clear error", "ollama.com/download" in str(e))
    finally:
        if old is None:
            os.environ.pop("OLLAMA_HOST", None)
        else:
            os.environ["OLLAMA_HOST"] = old
    print("llm selftest OK" if ok else "llm selftest FEIL / FAILED")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Lokal LLM via Ollama / Local LLM via Ollama")
    ap.add_argument("prompt", nargs="*")
    ap.add_argument("--model", help="Modellnavn, f.eks. qwen2.5-coder:7b / model name")
    ap.add_argument("--system", help="Systemprompt / system prompt")
    ap.add_argument("--models", action="store_true", help="Vis installerte modeller og anbefaling / show installed models and recommendation")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.models:
        return show_models()
    if not a.prompt:
        ap.print_help()
        return 0
    try:
        name, warn = pick_model(installed_models(), a.model)
        if warn:
            print(f"! {warn}", file=sys.stderr)
        print(f"[{name}]", file=sys.stderr)
        msgs = ([{"role": "system", "content": a.system}] if a.system else []) + [{"role": "user", "content": " ".join(a.prompt)}]
        chat(msgs, name, lambda t: (sys.stdout.write(t), sys.stdout.flush()))
        print()
    except OllamaError as e:
        print(f"FEIL / ERROR: {e}")
        return 1
    return 0


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    sys.exit(main())
