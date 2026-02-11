# TTS Expressive Tags (Rime Arcana & ElevenLabs v3)

Use expressive tags in your agent **personality_prompt**, **intro_phrase**, and LLM output to make speech livelier. **The correct tag list is automatically injected into the prompt passed to the model** based on `tts.provider` in the agent config (see `build_agent_instructions` in `rime_agent.py`). You do not need to copy the list into your config.

---

## Rime Arcana (angle brackets `<...>`)

Used when `tts.provider` is **rime**, **silero**, **kokoro**, or anything other than **elevenlabs**. Rime Arcana supports these natively; Silero/Kokoro may ignore them.

### Non-verbal / emotional (self-closing)

| Tag | Description | Example |
|-----|-------------|---------|
| `<laugh>` | Laughter | "Hey! <laugh> That's so funny." |
| `<chuckle>` | Light chuckle | "<chuckle> Yeah, I get it." |
| `<sigh>` | Sigh (empathy, tired, content) | "Aw, babe... <sigh> I'm here." |
| `<mmm>` | Humming / thinking | "<mmm> Good question. So..." |
| `<uh>` | Filler | "So, <uh> anyway." |
| `<um>` | Filler | "<um> Let me think." |
| `<clearthroat>` | Clear throat | "<clearthroat> As I was saying." |
| `<cough>` | Cough | "Sorry, <cough> allergies." |
| `<yawn>` | Yawn (relaxed, sleepy) | "It's late... <yawn> Anyway." |

### Wrappers (wrap the phrase to modify)

| Tag | Description | Example |
|-----|-------------|---------|
| `<whis>...</whis>` | Whisper (speak softly) | "<whis>I missed you.</whis>" |
| `<fast>...</fast>` | Faster speech | "<fast>Okay okay!</fast>" |
| `<slow>...</slow>` | Slower speech | "<slow>Listen carefully.</slow>" |
| `<pitch value="X">...</pitch>` | Adjust pitch (X = number) | "<pitch value=\"1.2\">Higher.</pitch>" |

---

## ElevenLabs v3 (square brackets `[...]`)

Used when `tts.provider` is **elevenlabs**. **These tags only work with the Eleven v3 model.** Earlier models (v2, Turbo) will ignore them or read the words literally.

### Laughter

- `[laughs]`, `[chuckle]`, `[giggles]`

### Sighing / breath

- `[sighs]`, `[exhales]`

### Thinking / fillers

- `[thinking]`, `[hmm]`, `[um]`

### Whispering

- `[whispers]` or `[whispering]` — place **before** the phrase you want whispered (no closing tag like Rime’s `<whis>...</whis>`).

### Pauses

- `[pause]`, `[short pause]`, `[long pause]`

### Other (v3 is flexible)

- `[shouting]`, `[crying]`, `[strong French accent]`, etc.

**Key difference:** ElevenLabs tags are usually “trigger” prompts before a phrase; there is no closing tag like Rime’s `<whis>...</whis>`.

---

## Guidelines

- **Don’t overuse.** One or two tags per reply is enough.
- **Match the character.** E.g. Ludia uses `<laugh>`, `<chuckle>`, `<sigh>` often; Osmond might use `<mmm>` or `<sigh>` sparingly.
- **Model requirement (ElevenLabs):** If you use an older ElevenLabs model (e.g. v2.5), it does not support bracketed tags; use punctuation or stability settings instead.

## In agent JSON

- **personality_prompt**: You can add style hints (e.g. “Use <laugh>, <sigh>, <mmm> when it fits”) and examples. The **full tag list is already injected** into the prompt sent to the model based on `tts.provider`.
- **greeting.intro_phrase**: e.g. `"Hey, I'm here. <laugh> What's on your mind?"` (use angle brackets for Rime/Silero, square brackets for ElevenLabs v3).
