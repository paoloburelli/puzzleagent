import requests
import json


class Simulator(object):

    def __init__(self, host: str, port: int, **kwargs):
        self.url = f"http://{host}:{port}"

    def load(self, level_index: int, seed: int) -> object:
        return self._do_request('/load', {
            'seed': seed,
            'levelIndex': level_index
        })

    def click(self, state: object, x, y) -> object:
        return self._do_request('/click', {
            'state': state,
            'x': x,
            'y': y
        })

    def session_create(self, level: int, seed: int, return_full_state: bool = False) -> any:
        return self._do_request('/session/create', {
            'levelIndex': level,
            'seed': seed,
            'returnFullState': return_full_state
        })

    def session_click(self, sessionId: str, x: int, y: int, dry_run=False, return_full_state: bool = False) -> object:
        try:
            result = self._do_request('/session/click', {
                'sessionId': sessionId,
                'x': x,
                'y': y,
                'returnFullState': return_full_state,
                'dryRun': dry_run
            })
        except requests.HTTPError as e:  # Workaround because error sometimes occurs
            result = self._do_request('/session/click', {
                'sessionId': sessionId,
                'x': x,
                'y': y,
                'returnFullState': return_full_state
            })

        return result

    def session_destroy(self, sessionId: str) -> object:
        return self._do_request('/session/destroy', {
            'sessionId': sessionId
        })

    def session_status(self, sessionId: str) -> object:
        return self._do_request('/session/status', {
            'sessionId': sessionId
        })

    def sessions_list(self) -> object:
        return self._do_request('/sessions/list', {})

    def sessions_clear(self) -> object:
        return self._do_request('/sessions/clear', {})

    def _do_request(self, path: str, json_payload: object) -> object:
        headers = {"content-type": "application/json"}
        response = requests.post(self.url + path, data=json.dumps(json_payload), headers=headers, timeout=100)
        response.raise_for_status()
        return response.json()
