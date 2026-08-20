"""ТОЛЬКО IPv4 для исходящих соединений процесса — одно место на весь бот.

⛔ ПОВОД, ЗАМЕРЕН НА ЭТОМ VPS 19.08.2026 (curl без ключа, только транспорт):
    IPv4 172.217.113.4        → ответ за 0.18 с
    IPv6 2001:4860:4842:400:: → 8 с молчания, код 000
То есть IPv6 к `generativelanguage.googleapis.com` с этого сервера — чёрная дыра. Резолвер
отдаёт AAAA первым, `requests` идёт по нему и висит до таймаута (в логе: `Read timed out
(read timeout=60)`, 15 раз за два часа). Каждое такое повисание — минута простоя прохода
extraction и ни одного отправленного запроса.

⛔ ПОЧЕМУ ЗДЕСЬ, А НЕ В КАЖДОМ ВЫЗОВЕ: в pseo эта же болезнь вылечена в `builder/keybroker.py`
ещё в июле, а во второе дерево починку не перенесли. Правило живёт ОДНИМ местом, иначе третий
вызывающий Gemini заведёт свою копию и снова будет висеть.

Фильтр процессный (правит `socket.getaddrinfo`), поэтому накрывает и `requests`, и SDK
`google-genai` (он ходит своим клиентом), и Telethon. Для Telegram это безопасно: у него есть
IPv4, а IPv6 на этом хосте всё равно не работает.

⚠️ ОТЛИЧИЕ ОТ ЭТАЛОНА В pseo: там фильтр отдаёт пустой список, если A-записей нет вовсе, —
на IPv6-only хосте это сломало бы резолв целиком. Здесь при отсутствии IPv4 возвращаем как
было: фильтр снимает ПРЕДПОЧТЕНИЕ IPv6, но не имеет права ломать разрешение имён.
"""

import socket

_orig_getaddrinfo = socket.getaddrinfo
_applied = False


def force_ipv4() -> bool:
    """Отфильтровать `getaddrinfo` до AF_INET. Идемпотентно.

    Returns:
        True — фильтр поставлен этим вызовом; False — уже стоял.
    """
    global _applied
    if _applied:
        return False

    def _ipv4_only(*args, **kwargs):
        res = _orig_getaddrinfo(*args, **kwargs)
        return [r for r in res if r[0] == socket.AF_INET] or res

    socket.getaddrinfo = _ipv4_only
    _applied = True
    return True
