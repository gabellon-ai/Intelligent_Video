"""Stream service for WebSocket/RTSP - stub"""


class StreamService:
    def list_streams(self):
        return []

    async def shutdown(self):
        pass


_instance = None


def init_stream_service(detector):
    global _instance
    _instance = StreamService()
    return _instance


def get_stream_service():
    return _instance
