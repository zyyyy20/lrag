import unittest

from app.routers.chat import router


class ChatRouteTests(unittest.TestCase):
    def test_non_streaming_chat_route_is_not_registered(self):
        post_paths = {
            route.path
            for route in router.routes
            if "POST" in getattr(route, "methods", set())
        }

        self.assertNotIn("/api/chat", post_paths)
        self.assertIn("/api/chat/stream", post_paths)


if __name__ == "__main__":
    unittest.main()
