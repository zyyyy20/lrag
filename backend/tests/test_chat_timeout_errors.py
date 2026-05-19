import unittest


class ChatTimeoutErrorTests(unittest.TestCase):
    def test_stream_error_message_is_readable_for_model_timeout(self):
        import httpx

        from app.routers.chat import _stream_error_message

        message = _stream_error_message(httpx.ReadTimeout("read timed out"))

        self.assertIn("模型响应超时", message)
        self.assertNotIn("httpx", message)


if __name__ == "__main__":
    unittest.main()
