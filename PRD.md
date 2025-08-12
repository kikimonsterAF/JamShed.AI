📄 Product Requirements Document (PRD)
Product Name: JamShed.AI (working title)
Prepared by: [Your Name]
Date: Aug 12, 2025
Version: 2.0

1. Product Overview
JamTab is a mobile app (iOS & Android) that uses AI to transcribe audio or video into tablature, chord progressions, and rhythm patterns. It supports beginner, intermediate, and advanced difficulty modes, offers practice tools like slow-down, looping, chunking, and reverse playback, and now includes song form detection (ABA, AABB, ABCABA, etc.) to show how the chord progression fits into the song’s structure.

Ultimate tool for shedding music.

2. Goals & Objectives
Provide highly accurate AI-driven transcription for solos, chords, and rhythms.

Display song structure labels for better learning and arrangement understanding.

Allow musicians to practice efficiently with advanced playback controls.

Remain genre-agnostic while giving Bluegrass-specific support.

3. Target Audience
Musicians of all skill levels

Bluegrass jammers & acoustic players

Music teachers & students

Transcribers & arrangers

4. Key Features
4.1 Audio/Video Input
Upload local audio/video or paste YouTube link.

Select timestamps for solos or full-song analysis.

4.2 Instrument Detection & Selection
AI suggests detected instrument; manual override available.

4.3 Solo Transcription
Beginner, Intermediate, Advanced tablature output.

Includes embellishments (hammer-ons, pull-offs, slides).

4.4 Chord Progression Analysis
Detect chords in standard notation.

Suggest capo/key.

Show Tennessee Number System.

4.5 Song Structure Detection (NEW)
Detect repeating sections & label form (A, B, C…).

Beginner view: simple ABA, AABB labels.

Advanced view: labels + bar counts + timestamps.

Clickable section letters to jump playback.

Loop-by-section capability.

4.6 Strumming Pattern Generator
Suggest patterns per difficulty level.

4.7 Practice Tools
Slow-down mode.

Loop sections.

Chunk practice mode.

Reverse playback mode.

4.8 Solo Breakdown Mode
Work backwards from last measure to first.

4.9 Scale Suggestions for Improvisation (NEW)
Suggest scales per chord and per song section to guide improvisation.

- Supported scales (initial): Minor Pentatonic, Major Pentatonic, Mixolydian, Dorian, Aeolian, Blues.
- Difficulty modes:
  - Beginner: show 1-2 safe options (usually pentatonic + blues when applicable).
  - Intermediate: add modal options (mixolydian over dominant, dorian over minor ii, aeolian for vi, etc.).
  - Advanced: include chord-tone targets, avoid notes, and passing tones per bar.
- Instrument-aware patterns: fretboard/position suggestions based on tuning and difficulty.
- Playback sync: highlight current section’s recommended scale and target tones as audio plays.
- Output: JSON with per-section and per-chord `key_center`, `recommended_scales`, `scale_degrees`, and `target_notes`.

4.10 Key Transpose While Practicing (NEW)
Allow users to change the playback key without re-uploading, for comfortable singing/playing and exploring tonalities.

- Real-time or near-real-time pitch shift of audio/video by ±12 semitones.
- Auto-transpose displayed chords, Tennessee numbers, and tablature to the selected key.
- Show capo suggestion for string instruments after transposition.
- Preserve tempo (independent of existing slow-down feature).
- Output/state: include `transpose_semitones` in session/practice settings and return transposed chord chart/tab.

5. Non-Functional Requirements
<30s transcription latency for 30s clip.

Smooth playback down to 25% speed.

Cross-platform support (iOS & Android).

6. Competitive Differentiators
Combines tab, chord progression, song form in one app.

Loop by form section (unique to this app).

Tailored to Bluegrass but works for any style.

7. Tech Requirements
Frontend: React Native / Flutter.

Backend: Node.js or Python (FastAPI) + AWS/GCP.

Analysis: Server-side only for MVP (no on-device processing). Mobile always uploads media to backend for analysis.

AI Models:

Whisper or equivalent for transcription.

CNN for instrument classification.

Custom chord & form detection model.

8. Roadmap
MVP: Input → Instrument selection → Tab + Chords + Form → Practice Tools.
Phase 2: Strumming patterns, advanced form view.
Phase 3: Cloud library & community features.

🛠 Development Task List
Phase 1 – Planning
Finalize PRD & wireframes.

Define AI model pipeline.

Plan legal compliance for music copyright.

Phase 2 – Backend Development
Cloud hosting setup.

Audio/video ingestion pipeline.

AI transcription engine for melody & chords.

NEW: Song form detection algorithm:

Segment detection via chord similarity.

Label assignment (A, B, C…).

JSON output with timestamps & bar counts.

Capo/key suggestion logic.

Strumming pattern generation.

API endpoints for all above.

Phase 3 – Frontend Development
Onboarding & login.

Home dashboard with upload/link input.

Timestamp preview & section selection.

Instrument & difficulty selection UI.

Transcription result screen with:

Tab view.

Chord progression view + form row at top.

Strumming pattern view.

Practice tools panel (slow, loop, chunk, reverse).

Solo breakdown mode.

Save & share workflow.

Library screen.

Phase 4 – Testing
Unit tests for AI outputs.

Integration tests for workflows.

Beta testing with musicians.

Performance testing at slow speeds.

Phase 5 – Launch Prep
App Store & Google Play submission.

Marketing assets.

Beta → public release.

📱 Updated Wireframe Flow
1. Onboarding
Welcome → Highlights → Sign Up/Login.

2. Home Dashboard
Upload file / paste link.

Recent projects carousel.

3. Preview & Timestamp Selection
Waveform/video preview.

Mark start/end for analysis.

4. Instrument & Difficulty Selection
AI suggestion + manual override.

5. Transcription Results
Tab View: Full tab synced with playback.

Chord Progression View:

Top: Song Form Row (A | B | A | B).

Middle: Chord chart synced to playback.

Bottom: Tennessee numbering + capo suggestion.

Strumming Pattern View: Visual arrows, beat diagrams.

Scale Suggestions View:
- Section header shows key center and active chord.
- List of recommended scales (by difficulty) with quick badges: Safe, Modal, Advanced.
- Fretboard/position helper for selected instrument.
- Tap to preview scale degrees against backing audio.

6. Practice Tools
Slow slider, loop points, chunk playback, reverse mode.

7. Solo Breakdown Mode
List of phrases with play/loop/reverse controls.

8. Save & Share
Save to cloud/local.

Export PDF with tab, chords, & form.

9. Library
Search, filter, resume projects.
 
 10. Monetization & Pricing
 Free Tier:
 - 5-song transcription quota per user (MVP; may switch to monthly reset later).
 - Access to all features within quota; optional watermark on exports.
 
 Subscription Tier (Paid):
 - Unlimited transcriptions and full feature access.
 - Pricing TBD; may include trials or intro offers.
 
 Purchase & Entitlements
 - Mobile IAP via App Store and Google Play. Consider RevenueCat for cross-platform receipts/entitlements and server-side webhooks.
 - Backend enforcement: track `transcriptions_used`; deny new jobs beyond quota unless active subscription.
 - Support restore purchases and handle grace periods/billing retry states.
 
 UX
 - Usage meter on Home and Results screens.
 - Paywall shown at limit or via Upgrade entry points.
 - Allow previously queued analyses to complete even if limit is reached mid-process.
 
 Analytics & Compliance
 - Measure paywall impressions, trials, conversions, churn.
 - Ensure IAP guideline compliance; update Terms/Privacy for subscriptions.