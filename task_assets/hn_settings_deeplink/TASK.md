# Task: Add Settings Deep Link and Capture Screenshot

## Objective

Add support for a custom URL scheme deep link `hackernews://settings` that navigates the app directly to the Settings screen, then capture a screenshot proving the implementation works.

## Requirements

### 1. Implement Deep Link Handler

Extend the existing URL scheme handling to support `hackernews://settings`:

- The app already has URL scheme `hackernews://` registered in the Info.plist
- The existing `handleDeepLink()` function in `HNApp.swift` handles `hackernews://story/{id}`
- Add support for `hackernews://settings` which should navigate to the Settings tab

### 2. Make TabView Selection Controllable

The current TabView in ContentView.swift doesn't expose programmatic tab selection. You'll need to:

- Add a way to track/control the selected tab (e.g., via AppViewModel)
- Update ContentView to use this selection state
- Update the deep link handler to set the tab to Settings when `hackernews://settings` is opened

### 3. Build, Install, and Launch via Deep Link

- Build the app for the iOS Simulator
- Install it on iPhone 17 Pro simulator
- Launch the app using the deep link: `xcrun simctl openurl booted "hackernews://settings"`

### 4. Capture Screenshot

After launching via the deep link, capture a screenshot proving the Settings screen is displayed:

**Important**:
- iOS may show a system confirmation dialog ("Open in HackerNews?") - this must be dismissed before taking the screenshot
- Wait for the app to fully load and display the Settings screen
- The screenshot must show the actual Settings screen UI (with "Settings" title, Profile section, etc.), NOT a system dialog

```bash
# Wait for app to load, then capture
sleep 3
xcrun simctl io booted screenshot ./settings_screenshot.png
```

The screenshot MUST be saved as `settings_screenshot.png` in the current working directory (the repo root).

**Note**: The grader validates that your screenshot shows the Settings screen by comparing it to a reference image. Screenshots showing system dialogs or other screens will fail validation.

## Constraints

- Do NOT modify the default app launch behavior - the app should still launch to the Feed tab by default
- Do NOT modify SettingsScreen.swift - the Settings UI should remain unchanged
- Only modify files necessary to add the deep link routing

## Success Criteria

1. The app builds successfully
2. The deep link `hackernews://settings` navigates to the Settings screen
3. A screenshot file `settings_screenshot.png` exists in the repo root
4. The screenshot shows the Settings screen (not the Feed or Bookmarks)
5. The app still launches to Feed by default (deep link doesn't change default behavior)

## Hints

- Look at how `handleDeepLink()` in HNApp.swift handles `hackernews://story/{id}`
- The TabView in ContentView.swift needs a `selection` binding to be controllable
- AppViewModel.swift is the right place to add tab selection state
- Use `Tab` enum values or integer indices for tab selection
