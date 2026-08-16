#!/usr/bin/env python3
"""Capture LinkedIn-ready screenshots from the locally running Dash app.

This script only connects to localhost: the Dash app on port 8050 and a local
Chrome debugging session on port 9223. It does not use the internet.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import socket
import struct
import time
from urllib.request import urlopen


CHROME_DEBUG_URL = "http://127.0.0.1:9223/json"
APP_URL = "http://127.0.0.1:8050"
OUTPUT_DIR = Path(__file__).resolve().parent / "screenshots"


class ChromeDevTools:
    """Minimal WebSocket client for the small CDP surface used here."""

    def __init__(self, websocket_url: str) -> None:
        scheme, remainder = websocket_url.split("://", 1)
        if scheme != "ws":
            raise ValueError("Only local ws:// Chrome endpoints are supported")
        authority, path = remainder.split("/", 1)
        host, port_text = authority.rsplit(":", 1)
        self.socket = socket.create_connection((host, int(port_text)), timeout=30)
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        request = (
            f"GET /{path} HTTP/1.1\r\n"
            f"Host: {authority}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        )
        self.socket.sendall(request.encode("ascii"))
        response = self._read_until(b"\r\n\r\n")
        if b" 101 " not in response.split(b"\r\n", 1)[0]:
            raise RuntimeError(f"Chrome WebSocket handshake failed: {response!r}")
        self.next_id = 1

    def _read_until(self, marker: bytes) -> bytes:
        data = b""
        while marker not in data:
            chunk = self.socket.recv(4096)
            if not chunk:
                raise ConnectionError("Chrome closed the debugging connection")
            data += chunk
        return data

    def _read_exact(self, length: int) -> bytes:
        data = b""
        while len(data) < length:
            chunk = self.socket.recv(length - len(data))
            if not chunk:
                raise ConnectionError("Chrome closed the debugging connection")
            data += chunk
        return data

    def _send_frame(self, payload: bytes, opcode: int = 1) -> None:
        mask = os.urandom(4)
        length = len(payload)
        header = bytearray([0x80 | opcode])
        if length < 126:
            header.append(0x80 | length)
        elif length < 65536:
            header.append(0x80 | 126)
            header.extend(struct.pack("!H", length))
        else:
            header.append(0x80 | 127)
            header.extend(struct.pack("!Q", length))
        masked = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
        self.socket.sendall(bytes(header) + mask + masked)

    def _read_message(self) -> str:
        fragments = bytearray()
        started = False
        while True:
            first, second = self._read_exact(2)
            final = bool(first & 0x80)
            opcode = first & 0x0F
            length = second & 0x7F
            if length == 126:
                length = struct.unpack("!H", self._read_exact(2))[0]
            elif length == 127:
                length = struct.unpack("!Q", self._read_exact(8))[0]
            masked = bool(second & 0x80)
            mask = self._read_exact(4) if masked else b""
            payload = self._read_exact(length)
            if masked:
                payload = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
            if opcode == 9:
                self._send_frame(payload, opcode=10)
                continue
            if opcode == 8:
                raise ConnectionError("Chrome closed the debugging connection")
            if opcode in (1, 2):
                fragments = bytearray(payload)
                started = True
            elif opcode == 0 and started:
                fragments.extend(payload)
            if final and started:
                return fragments.decode("utf-8")

    def call(self, method: str, params: dict | None = None) -> dict:
        request_id = self.next_id
        self.next_id += 1
        self._send_frame(
            json.dumps({"id": request_id, "method": method, "params": params or {}}).encode("utf-8")
        )
        while True:
            response = json.loads(self._read_message())
            if response.get("id") != request_id:
                continue
            if "error" in response:
                raise RuntimeError(f"CDP {method} failed: {response['error']}")
            return response.get("result", {})

    def evaluate(self, expression: str) -> object:
        result = self.call(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True, "awaitPromise": True},
        )
        remote = result.get("result", {})
        if "exceptionDetails" in result:
            raise RuntimeError(result["exceptionDetails"])
        return remote.get("value")


def wait_for(devtools: ChromeDevTools, expression: str, timeout: int = 45) -> object:
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = devtools.evaluate(expression)
        if value:
            return value
        time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for browser expression: {expression}")


def capture_target(
    devtools: ChromeDevTools,
    filename: str,
    target_expression: str,
    max_height: int = 1300,
) -> None:
    found = devtools.evaluate(
        f"""
        (() => {{
          const target = {target_expression};
          if (!target) return false;
          window.__linkedinCaptureTarget = target;
          target.scrollIntoView({{block: 'start', inline: 'nearest'}});
          window.scrollBy(0, -18);
          return true;
        }})()
        """
    )
    if not found:
        raise RuntimeError(f"Could not find screenshot target for {filename}")
    time.sleep(1.2)
    rect = devtools.evaluate(
        """
        (() => {
          const rect = window.__linkedinCaptureTarget.getBoundingClientRect();
          const padding = 12;
          return {
            x: Math.max(0, rect.left + window.scrollX - padding),
            y: Math.max(0, rect.top + window.scrollY - padding),
            width: Math.min(document.documentElement.scrollWidth, rect.width + padding * 2),
            height: rect.height + padding * 2
          };
        })()
        """
    )
    rect["height"] = min(rect["height"], max_height)
    screenshot = devtools.call(
        "Page.captureScreenshot",
        {
            "format": "png",
            "captureBeyondViewport": True,
            "fromSurface": True,
            "clip": {**rect, "scale": 1},
        },
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / filename).write_bytes(base64.b64decode(screenshot["data"]))
    print(f"Captured {filename}: {round(rect['width'])} x {round(rect['height'])}")


def heading_section(title: str) -> str:
    encoded = json.dumps(title)
    return (
        "(() => {"
        f"const heading=[...document.querySelectorAll('h3')].find(node => node.textContent.trim()==={encoded});"
        "return heading ? (heading.closest('section') || heading.closest('.content-section')) : null;"
        "})()"
    )


def main() -> None:
    with urlopen(CHROME_DEBUG_URL, timeout=10) as response:
        pages = json.load(response)
    page = next((item for item in pages if item.get("type") == "page"), None)
    if not page:
        raise RuntimeError("No Chrome page target is available on local debugging port 9223")

    devtools = ChromeDevTools(page["webSocketDebuggerUrl"])
    devtools.call("Page.enable")
    devtools.call("Runtime.enable")
    devtools.call(
        "Emulation.setDeviceMetricsOverride",
        {"width": 1440, "height": 1100, "deviceScaleFactor": 1, "mobile": False},
    )
    devtools.call("Page.navigate", {"url": APP_URL})
    wait_for(devtools, "document.readyState === 'complete' && !!document.querySelector('#question-mode')")
    time.sleep(2)

    capture_target(
        devtools,
        "00-sales-manager-overview.png",
        "document.querySelector('.manager-portal')",
        1050,
    )

    selected = devtools.evaluate(
        """
        (() => {
          const option = document.querySelector('#question-mode input[value="llm"]');
          if (!option || option.disabled) return false;
          option.click();
          return true;
        })()
        """
    )
    if not selected:
        raise RuntimeError("The Local LLM option is not available in the running app")

    question = (
        "For Alice Brown, summarise the top three customer whitespace opportunities by "
        "estimated annual potential and explain the recorded next conversation for each."
    )
    submitted = devtools.evaluate(
        f"""
        (() => {{
          const input = document.querySelector('#data-question-input');
          const button = document.querySelector('#ask-data-button');
          if (!input || !button) return false;
          const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
          setter.call(input, {json.dumps(question)});
          input.dispatchEvent(new Event('input', {{bubbles: true}}));
          input.dispatchEvent(new Event('change', {{bubbles: true}}));
          button.click();
          return true;
        }})()
        """
    )
    if not submitted:
        raise RuntimeError("Could not submit the Local LLM question")
    wait_for(
        devtools,
        """
        (() => {
          const title = document.querySelector('#data-answer .answer-title');
          return title && title.textContent.includes('Local LLM: Customer whitespace opportunities');
        })()
        """,
        timeout=150,
    )
    time.sleep(1)
    capture_target(devtools, "01-local-llm-question.png", "document.querySelector('.ask-section')", 1200)
    capture_target(
        devtools,
        "05-sales-person-overview.png",
        "document.querySelector('#salesperson-detail-section')",
        1300,
    )

    clicked = devtools.evaluate(
        """
        (() => {
          const candidates = [...document.querySelectorAll('[role="tab"], .tab, .custom-tab')];
          const tab = candidates.find(node => node.textContent.trim() === 'Model Analysis');
          if (!tab) return false;
          tab.click();
          return true;
        })()
        """
    )
    if not clicked:
        raise RuntimeError("Could not find the Model Analysis tab")
    wait_for(devtools, "!!document.querySelector('.data-scientist-portal')")
    time.sleep(2.5)

    capture_target(
        devtools,
        "02-revenue-model-purpose.png",
        heading_section("What the model predicts and how to interpret it"),
        1150,
    )
    capture_target(
        devtools,
        "03-model-validation.png",
        heading_section("Candidate model leaderboard"),
        1250,
    )
    capture_target(
        devtools,
        "04-feature-impact.png",
        heading_section("Feature behaviour"),
        1250,
    )


if __name__ == "__main__":
    main()
