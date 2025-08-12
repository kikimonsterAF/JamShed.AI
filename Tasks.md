 Development Task List
Phase 1 – Planning & Design
Finalize PRD with all features including song form detection (ABA, AABB, ABCABA).

Approve UI Wireframes & Flow with new chord progression form row.

Define Technical Architecture

Choose mobile framework (React Native / Flutter).

Choose backend stack (Node.js or Python FastAPI).

Select AI transcription model(s).

Legal Compliance

Review copyright implications of transcription & YouTube usage.

Prepare privacy policy & terms of service.

Monetization Planning
- Define free quota policy (MVP: 5-song lifetime; evaluate monthly reset later).
- Choose IAP/entitlements stack (native StoreKit/Billing vs. RevenueCat recommended).
- Define product IDs, entitlement names, and paywall copy/FAQ.

Phase 2 – Backend Development
Infrastructure
Set up cloud environment (AWS/GCP/Azure).

Configure storage for audio/video (e.g., AWS S3).

Media Handling
Implement upload handling for local files & YouTube/Vimeo links.

Extract audio from video sources.

Normalize audio levels for analysis.

AI & Analysis Engine
Transcription Module

Implement melody detection for solos.

Map detected notes to tablature for selected instrument.

Instrument Detection Module

Build/train CNN classifier for instrument recognition.

Chord Progression Module

Chord detection (symbol, Tennessee numbering).

Key & capo suggestion.

Song Structure Detection Module (NEW)

Segment song by measures.

Identify repeating patterns.

Assign section labels (A, B, C).

Output JSON with section letters, bar counts, and timestamps.

Strumming Pattern Generator

Extract rhythm patterns from chord track.

Generate beginner/intermediate/advanced patterns.

Scale Suggestions Module (NEW)
- Derive key centers per section from detected chords and global key.
- Map chords to recommended scales by function (e.g., V7 → Mixolydian; minor → Dorian/Aeolian; universal safe → Minor Pentatonic/Blues).
- Provide difficulty-tiered lists (Beginner/Intermediate/Advanced) and annotate scale degrees/avoid notes.
- Instrument-aware: translate to fretboard positions per tuning.
- JSON schema: per-section and per-chord recommendations with timestamps.

Monetization & Entitlements
- Database: add `subscription_status`, `transcriptions_used`, `last_reset_at`.
- Quota enforcement middleware on transcription endpoints.
- Subscription webhooks (RevenueCat or store server notifications).
- Receipt validation and purchase restore endpoint.
- Usage meter endpoint (return remaining quota and entitlement status).

API Development
Authentication (OAuth, email, Apple ID, Google).

Endpoints for:

Uploads & transcription requests.

Tab, chord, and structure data retrieval.

Practice tool settings (loop, slow-down, reverse).

Phase 3 – Frontend Development
Onboarding
Welcome carousel with feature highlights.

Sign-up/login integration.

Home Dashboard
File upload & link paste UI.

Recent projects carousel.

Preview & Timestamp Selection
Waveform/video preview with start/end selection.

Instrument & Difficulty Selection
AI suggestion display with manual override.

Beginner, Intermediate, Advanced toggle.

Transcription Results
Tab View — scrollable tablature synced with audio/video.

Chord Progression View —

Song Form Row at top (A | B | A | B).

Chords in standard & Tennessee numbering.

Capo suggestion.

Clickable section letters for playback navigation.

Strumming Pattern View —

Visual D/U arrows.

Beginner/intermediate/advanced variations.

Scale Suggestions View —
- UI to show recommended scales per section/chord (badges: Safe/Modal/Advanced).
- Fretboard helper overlay; tap to audition scale degrees.
- Sync highlight with playback position.

Monetization UI
- Usage meter component on Home and Results screens.
- Paywall screen with feature list and legal links.
- Purchase flow integration (StoreKit/Billing via RevenueCat SDK).
- Restore purchases action.
- Entitlement hydration from backend at app start and after purchases.

Practice Tools
Slow-down slider (100% → 25%).

Loop start/end selection.

Chunk practice mode.

Reverse playback toggle.

Solo Breakdown Mode
Phrase list view with play/loop/reverse buttons.

Work backwards mode.

Save & Share
Save to cloud/local.

Export PDF with tabs, chords, song form.

Library
Project grid with search & filter.

Phase 4 – Testing & QA
Unit Tests — AI model outputs, API endpoints.

Integration Tests — Full upload-to-results flow.

User Testing — Beginner, intermediate, advanced musicians.

Performance Tests — Latency, playback smoothness.

Phase 5 – Launch Prep
App Store & Google Play developer accounts.

Finalize marketing materials (screenshots, promo video).

Beta release to closed group.

Collect feedback & fix issues.

Full public release.

Phase 6 – Post-Launch Enhancements
Community tab sharing.

Genre presets.

AI jam track generation.

Desktop/web companion version.