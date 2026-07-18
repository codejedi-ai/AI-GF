# Integration Brainstorm: Galatea Calling (Matrix, Telegram, Discord)

This document outlines the technical paths for bridging the Galatea-LiveKit agent into external communication platforms.

---

## 1. Telegram (Recommended Path)
Telegram offers the best user experience because it allows embedding a LiveKit web client directly within the app.

### **Path A: Telegram Mini App (TWA)**
*   **Concept**: Host a simple React/Next.js frontend using `livekit-components-react`. Use a Telegram Bot to launch this as a "Mini App."
*   **Pros**: 
    *   Full WebRTC support (lowest latency).
    *   Native feel; users don't leave Telegram.
    *   Full access to visuals (avatars, waveform).
*   **Tech Stack**: Telegram Bot API + LiveKit Web SDK + Vercel/similar for hosting.

### **Path B: Telegram Voice Chat Bridge**
*   **Concept**: A bot using `pyrogram` or `telethon` joins a Group Voice Chat and pipes audio back and forth to LiveKit.
*   **Pros**: Works in standard group calls.
*   **Cons**: Extremely difficult audio handling; Telegram's VoIP protocol is not natively compatible with WebRTC without heavy transcoding.

---

## 2. Discord
Discord is a closed ecosystem, so we must act as a "User" or "Bot" in a Voice Channel.

### **The "Audio Bridge" Bot**
*   **Concept**: A Discord bot joins a Voice Channel. It captures raw PCM audio from specific users, sends it to the Galatea Agent via a LiveKit `AudioTrack`, and vice versa.
*   **Pros**: Users stay in Discord.
*   **Cons**: 
    *   High Latency: Audio travels `User -> Discord -> Bridge Bot -> LiveKit -> Agent`.
    *   Compute Heavy: The server must handle multiple audio streams simultaneously.
*   **Tech Stack**: `discord.py` (with voice support) + `livekit-server-sdk`.

---

## 3. Matrix
Matrix is uniquely suited for LiveKit because its flagship calling implementation is built on LiveKit.

### **Path A: Native Element Call Integration**
*   **Concept**: **Element Call** uses LiveKit under the hood. Since Natasha is already a LiveKit agent, she can be invited directly to an Element Call session if given the room URL/token.
*   **Pros**: Zero transcoding; native quality and latency.
*   **Cons**: Requires the Matrix homeserver to be using the LiveKit-based Element Call stack.
*   **Tech Stack**: Matrix Rust SDK + LiveKit Python SDK.

### **Path B: Matrix-SIP Bridge**
*   **Concept**: Use LiveKit’s **SIP Ingress** to treat Matrix calls like a SIP phone call.
*   **Pros**: Highly standardized.
*   **Cons**: Visuals are lost; audio only.

---

## Technical Feasibility Summary

| Platform | Best Method | Difficulty | Latency |
| :--- | :--- | :--- | :--- |
| **Telegram** | Mini App (Web) | Low | Low |
| **Discord** | Audio Bridge Bot | High | Medium |
| **Matrix** | Element Call Native | Medium | Low |

## Next Steps
1.  **Telegram**: The fastest "WOW" factor. I can help you draft a simple `bot.py` that sends a "Call Natasha" button which opens a LiveKit room.
2.  **Discord**: Requires a dedicated "Transcoder" script.
3.  **Matrix**: Check if your Matrix server supports Element Call (LiveKit-based).
