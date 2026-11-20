# `cohub.federation` — several processes, one session

Each headset needs its own Isaac Sim process. The recording process (primary)
runs the physics and sends the scene's state to each mirror over UDP; a mirror
shows it and sends back its robot's 32D action.

| module | what it holds |
|---|---|
| `link.py` | the two datagram formats (state to a mirror, action back), the non-blocking UDP endpoints, and the relay that passes a reset between processes once |
| `scene_state.py` | reading the scene on the primary and re-posing it on a mirror |

The link never blocks: each side drains its socket and keeps the newest packet.
A lost datagram is tolerated, so anything that must arrive is sent as a level or
a counter, never as a one-shot event.
