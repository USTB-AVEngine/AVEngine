"""AVEngine Studio: the engine's web planning console.

Session 1 ships the backend skeleton: a loopback-only stdlib HTTP server
exposing read-only catalog APIs (rooms, registries, review captures) and a
serial subprocess render queue that wraps the engine's own CLI verbs. The
Studio never renders outside the engine chain and never widens data
admission: everything it queues is research_only with the formal dataset
denominator at 0.
"""
