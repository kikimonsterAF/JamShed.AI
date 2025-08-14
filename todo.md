# JamTab TODO List

## In Progress
- [ ] **ii-V-I Detection** - Implement ii-V-I progression detection
  - ✅ Added bluegrass major II support
  - ✅ Key detection fixed
  - 🔄 Need to debug progression analysis execution (Dm->D promotion not working)

## Pending Tasks
- [ ] **vi-ii-V-I Detection** - Implement vi-ii-V-I (extended) progression detection for jazz and sophisticated pop
- [ ] **Chord Function Analysis** - Add Roman numeral function analysis and progression strength scoring
- [ ] **Resolution Preference** - Bias chord detection toward common resolutions when ambiguous
- [ ] **Test Jazz Standards** - Validate with jazz standards and sophisticated pop songs

## Current Status
- Working on `polished-chords` branch
- AISU chord detection implemented with bluegrass progression analysis
- Key detection working correctly (returns "C major")
- Issue: Dm-G-C pattern not being promoted to D-G-C despite logic being in place

## Next Steps
1. Debug why `_analyze_progression_patterns_aisu` is not executing the Dm->D promotion
2. Investigate `_chord_matches_function` logic
3. Test progression analysis with more bluegrass songs
4. Expand to vi-ii-V-I patterns
