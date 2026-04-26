from __future__ import annotations

import struct
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ProbeResult:
    status: str = "disabled"
    player_world_pos: tuple[float, float, float] | None = None
    player_heading_deg: float | None = None
    map_open: bool | None = None
    biome_or_scene_id: str | None = None
    scene_id: str | None = None
    active_room_or_node_id: str | None = None
    room_start: tuple[float, float, float] | None = None
    room_end: tuple[float, float, float] | None = None
    is_in_crypt: bool | None = None
    graveyard_crypt_keys: int | None = None
    current_objective: str | None = None
    boss_spotted: bool | None = None
    charged_shrines: int | None = None
    ts: float = 0.0
    source: str = "external_memory"
    details: dict[str, Any] = field(default_factory=dict)


class WorldStateProbe:
    def sample(self, now_ts: float | None = None) -> ProbeResult:
        raise NotImplementedError

    def close(self) -> None:
        return None


class NullProbe(WorldStateProbe):
    def sample(self, now_ts: float | None = None) -> ProbeResult:
        now_ts = time.time() if now_ts is None else float(now_ts)
        return ProbeResult(status="disabled", ts=now_ts, source="external_memory")


class ExternalProcessProbe(WorldStateProbe):
    PROCESS_QUERY_INFORMATION = 0x0400
    PROCESS_VM_READ = 0x0010

    def __init__(
        self,
        *,
        window_title: str,
        poll_interval_s: float = 0.25,
        signatures: dict[str, Any] | None = None,
    ):
        self.window_title = str(window_title)
        self.poll_interval_s = max(0.05, float(poll_interval_s))
        self.signatures = dict(signatures or {})
        self._kernel32 = None
        self._user32 = None
        self._handle = None
        self._pid = 0
        self._gameassembly_module = None
        self._last_poll_ts = 0.0
        self._last_result = ProbeResult(status="cold_start", ts=0.0)
        try:
            import ctypes
            from ctypes import wintypes

            self._ctypes = ctypes
            self._wintypes = wintypes
            self._kernel32 = ctypes.windll.kernel32
            self._user32 = ctypes.windll.user32
        except Exception:
            self._ctypes = None
            self._wintypes = None
            self._kernel32 = None
            self._user32 = None

    def close(self) -> None:
        if self._handle and self._kernel32 is not None:
            try:
                self._kernel32.CloseHandle(self._handle)
            except Exception:
                pass
        self._handle = None

    def sample(self, now_ts: float | None = None) -> ProbeResult:
        now_ts = time.time() if now_ts is None else float(now_ts)
        if self._kernel32 is None or self._user32 is None or self._ctypes is None:
            self._last_result = ProbeResult(status="unsupported_platform", ts=now_ts)
            return self._last_result
        if (now_ts - self._last_poll_ts) < self.poll_interval_s:
            return self._last_result
        self._last_poll_ts = now_ts
        try:
            if not self._ensure_handle():
                self._last_result = ProbeResult(status="process_not_found", ts=now_ts)
                return self._last_result
            if not self.signatures:
                self._last_result = ProbeResult(
                    status="degraded_no_signatures",
                    ts=now_ts,
                    details={"pid": self._pid},
                )
                return self._last_result
            if self._gameassembly_module is None:
                self._gameassembly_module = self._find_module("GameAssembly.dll")
            if self._gameassembly_module is None:
                self._last_result = ProbeResult(
                    status="degraded_no_gameassembly",
                    ts=now_ts,
                    details={"pid": self._pid},
                )
                return self._last_result

            values: dict[str, Any] = {}
            for name, spec in self.signatures.items():
                try:
                    values[name] = self._read_signature_value(spec)
                except Exception as exc:
                    values[f"{name}_error"] = str(exc)

            ready_keys = (
                "player_world_pos",
                "player_heading_deg",
                "map_open",
                "biome_or_scene_id",
                "scene_id",
                "active_room_or_node_id",
                "room_start",
                "room_end",
                "is_in_crypt",
                "graveyard_crypt_keys",
                "current_objective",
                "boss_spotted",
                "charged_shrines",
            )
            self._last_result = ProbeResult(
                status="ready" if any(values.get(key) is not None for key in ready_keys) else "degraded_empty_read",
                player_world_pos=values.get("player_world_pos"),
                player_heading_deg=values.get("player_heading_deg"),
                map_open=values.get("map_open"),
                biome_or_scene_id=values.get("biome_or_scene_id"),
                scene_id=values.get("scene_id"),
                active_room_or_node_id=values.get("active_room_or_node_id"),
                room_start=values.get("room_start"),
                room_end=values.get("room_end"),
                is_in_crypt=values.get("is_in_crypt"),
                graveyard_crypt_keys=values.get("graveyard_crypt_keys"),
                current_objective=values.get("current_objective"),
                boss_spotted=values.get("boss_spotted"),
                charged_shrines=values.get("charged_shrines"),
                ts=now_ts,
                details={"pid": self._pid, **{k: v for k, v in values.items() if k.endswith("_error")}},
            )
            return self._last_result
        except Exception as exc:
            self._last_result = ProbeResult(
                status="degraded_error",
                ts=now_ts,
                details={"error": str(exc)},
            )
            return self._last_result

    def _ensure_handle(self) -> bool:
        pid = self._find_pid_by_window_title(self.window_title)
        if not pid:
            self.close()
            self._pid = 0
            self._gameassembly_module = None
            return False
        if pid == self._pid and self._handle:
            return True
        self.close()
        self._pid = pid
        self._gameassembly_module = None
        access = self.PROCESS_QUERY_INFORMATION | self.PROCESS_VM_READ
        self._handle = self._kernel32.OpenProcess(access, False, pid)
        return bool(self._handle)

    def _find_pid_by_window_title(self, title: str) -> int:
        hwnd = self._user32.FindWindowW(None, title)
        if not hwnd:
            return 0
        pid = self._wintypes.DWORD()
        self._user32.GetWindowThreadProcessId(hwnd, self._ctypes.byref(pid))
        return int(pid.value)

    def _find_module(self, module_name: str):
        TH32CS_SNAPMODULE = 0x00000008
        TH32CS_SNAPMODULE32 = 0x00000010

        class MODULEENTRY32W(self._ctypes.Structure):
            _fields_ = [
                ("dwSize", self._wintypes.DWORD),
                ("th32ModuleID", self._wintypes.DWORD),
                ("th32ProcessID", self._wintypes.DWORD),
                ("GlblcntUsage", self._wintypes.DWORD),
                ("ProccntUsage", self._wintypes.DWORD),
                ("modBaseAddr", self._ctypes.POINTER(self._ctypes.c_byte)),
                ("modBaseSize", self._wintypes.DWORD),
                ("hModule", self._wintypes.HMODULE),
                ("szModule", self._wintypes.WCHAR * 256),
                ("szExePath", self._wintypes.WCHAR * 260),
            ]

        snapshot = self._kernel32.CreateToolhelp32Snapshot(
            TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32,
            self._pid,
        )
        if snapshot == self._wintypes.HANDLE(-1).value:
            return None
        try:
            entry = MODULEENTRY32W()
            entry.dwSize = self._ctypes.sizeof(MODULEENTRY32W)
            if not self._kernel32.Module32FirstW(snapshot, self._ctypes.byref(entry)):
                return None
            while True:
                if str(entry.szModule).lower() == module_name.lower():
                    return {
                        "base_addr": self._ctypes.addressof(entry.modBaseAddr.contents),
                        "size": int(entry.modBaseSize),
                        "path": str(entry.szExePath),
                    }
                if not self._kernel32.Module32NextW(snapshot, self._ctypes.byref(entry)):
                    break
            return None
        finally:
            self._kernel32.CloseHandle(snapshot)

    def _read_process(self, address: int, size: int) -> bytes:
        if not self._handle:
            raise RuntimeError("Process handle is not available")
        buffer = (self._ctypes.c_ubyte * int(size))()
        bytes_read = self._wintypes.SIZE_T()
        ok = self._kernel32.ReadProcessMemory(
            self._handle,
            self._ctypes.c_void_p(int(address)),
            buffer,
            int(size),
            self._ctypes.byref(bytes_read),
        )
        if not ok:
            raise RuntimeError(f"ReadProcessMemory failed at 0x{int(address):X}")
        return bytes(buffer[: int(bytes_read.value)])

    @staticmethod
    def _parse_pattern(raw: str) -> list[int | None]:
        parts = str(raw).split()
        result: list[int | None] = []
        for token in parts:
            if token in {"?", "??"}:
                result.append(None)
            else:
                result.append(int(token, 16))
        return result

    def _scan_pattern(self, module: dict[str, Any], pattern: str) -> int | None:
        needle = self._parse_pattern(pattern)
        if not needle:
            return None
        raw = self._read_process(int(module["base_addr"]), int(module["size"]))
        limit = len(raw) - len(needle) + 1
        for offset in range(max(0, limit)):
            for idx, value in enumerate(needle):
                if value is None:
                    continue
                if raw[offset + idx] != value:
                    break
            else:
                return int(module["base_addr"]) + offset
        return None

    def _read_pointer(self, address: int) -> int:
        raw = self._read_process(address, 8)
        return int.from_bytes(raw[:8], byteorder="little", signed=False)

    def _resolve_address(self, spec: dict[str, Any]) -> int:
        address = spec.get("absolute_address")
        if address is None:
            pattern = spec.get("pattern")
            if not pattern:
                raise ValueError("Signature spec must contain absolute_address or pattern")
            address = self._scan_pattern(self._gameassembly_module, str(pattern))
            if address is None:
                raise ValueError(f"Pattern not found: {pattern}")
            address += int(spec.get("pattern_offset", 0))
        address = int(address)
        pointer_chain = spec.get("pointer_chain")
        if isinstance(pointer_chain, list):
            current = address
            for offset in pointer_chain:
                current = self._read_pointer(current + int(offset))
                if current <= 0:
                    raise ValueError("Pointer chain resolved to null")
            address = current
        return address

    @staticmethod
    def _decode_cstring(raw: bytes) -> str:
        return raw.split(b"\x00", 1)[0].decode("utf-8", errors="ignore").strip()

    @staticmethod
    def _decode_utf16(raw: bytes) -> str:
        text = raw.decode("utf-16-le", errors="ignore")
        return text.split("\x00", 1)[0].strip()

    def _read_signature_value(self, spec: dict[str, Any]) -> Any:
        if not isinstance(spec, dict):
            raise ValueError("Signature spec must be a dict")
        address = self._resolve_address(spec)
        read_type = str(spec.get("type", "u32")).lower()
        size_map = {
            "bool": 1,
            "u8": 1,
            "u32": 4,
            "i32": 4,
            "f32": 4,
            "f32x3": 12,
        }
        if read_type in size_map:
            raw = self._read_process(int(address), size_map[read_type])
            if read_type == "bool":
                value = bool(raw[0])
            elif read_type == "u8":
                value = int(raw[0])
            elif read_type == "u32":
                value = int.from_bytes(raw[:4], byteorder="little", signed=False)
            elif read_type == "i32":
                value = int.from_bytes(raw[:4], byteorder="little", signed=True)
            elif read_type == "f32":
                value = struct.unpack("<f", raw[:4])[0]
            elif read_type == "f32x3":
                value = tuple(float(item) for item in struct.unpack("<fff", raw[:12]))
            else:
                raise ValueError(f"Unsupported read type: {read_type}")
        elif read_type in {"ascii", "utf16"}:
            max_length = int(spec.get("max_length", 96))
            if max_length <= 0:
                raise ValueError("max_length must be > 0 for string reads")
            raw = self._read_process(int(address), max_length * (2 if read_type == "utf16" else 1))
            value = self._decode_utf16(raw) if read_type == "utf16" else self._decode_cstring(raw)
        else:
            raise ValueError(f"Unsupported read type: {read_type}")

        enum_map = spec.get("map")
        if isinstance(enum_map, dict):
            mapped = enum_map.get(str(value))
            if mapped is None:
                mapped = enum_map.get(value)
            if mapped is not None:
                return mapped
        return value
