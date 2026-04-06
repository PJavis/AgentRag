# Achievement System — Architecture & Specification

System design document for the Pamlogix Achievement System running on Nakama.

---

## Overview

The achievement system is a **Pamlogix module** (`AchievementsSystem`) that runs as a Go library inside a Nakama game server plugin. It provides:

- Progress tracking with count-based thresholds
- Hierarchical achievements (parent + sub-achievements)
- Repeatable and one-off achievements
- CRON-based automatic resets (daily, weekly, custom)
- Time-limited achievements (start/end/duration)
- Precondition chains (achievement A requires achievement B)
- Reward rolling and granting via the Economy system
- Pluggable reward hooks for game-specific side effects
- Auto-claim and auto-reset behaviors

```
┌──────────────────────────────────────────────────────┐
│  Game Plugin (Go)                                    │
│                                                      │
│  ┌─────────────┐   ┌──────────────┐                 │
│  │ Quest Module │   │ Coop Module  │  ... other      │
│  └──────┬──────┘   └──────┬───────┘      modules    │
│         │                 │                          │
│         ▼                 ▼                          │
│  ┌─────────────────────────────────────────────┐    │
│  │         AchievementsSystem (Pamlogix)        │    │
│  │  UpdateAchievements / ClaimAchievements      │    │
│  │  GetAchievements / MarkAsClaimed             │    │
│  │                                              │    │
│  │  ┌──────────────┐  ┌─────────────────────┐  │    │
│  │  │ Reward Hooks  │  │ EconomySystem       │  │    │
│  │  │ (pluggable)   │  │ RewardRoll + Grant  │  │    │
│  │  └──────────────┘  └─────────────────────┘  │    │
│  └───────────────────────┬─────────────────────┘    │
│                          │                          │
└──────────────────────────┼──────────────────────────┘
                           │ StorageRead / StorageWrite
                    ┌──────▼──────┐
                    │   Nakama    │
                    │  Storage    │
                    │ (CockroachDB)│
                    └─────────────┘
```

---

## Data Models

### Achievement (User State)

Each achievement tracked per-user has the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique achievement identifier |
| `category` | string | Grouping category for filtering |
| `name` | string | Display name (i18n key) |
| `description` | string | Description text (i18n key) |
| `count` | int64 | Current progress count |
| `max_count` | int64 | Completion threshold |
| `claim_time_sec` | int64 | Unix timestamp when claimed (0 = unclaimed) |
| `total_claim_time_sec` | int64 | Unix timestamp when total reward claimed |
| `current_time_sec` | int64 | Server timestamp at last update |
| `reset_time_sec` | int64 | Next scheduled reset time |
| `expire_time_sec` | int64 | Expiration time (0 = never) |
| `start_time_sec` | int64 | Start availability time |
| `end_time_sec` | int64 | End availability time |
| `precondition_ids` | []string | Required predecessor achievement IDs |
| `reward` | Reward | Rolled individual reward |
| `total_reward` | Reward | Rolled total completion reward |
| `available_rewards` | AvailableRewards | Reward definition with probabilities |
| `available_total_reward` | AvailableRewards | Total reward definition |
| `sub_achievements` | map[string]SubAchievement | Nested sub-achievements |
| `additional_properties` | map[string]string | Custom game-specific metadata |
| `auto_claim` | bool | Auto-claim when count reaches max_count |
| `auto_claim_total` | bool | Auto-claim total reward when all subs complete |
| `auto_reset` | bool | Reset count after claim |

### SubAchievement

Sub-achievements share most fields with the parent but do not have `sub_achievements` or `total_reward`. They represent individual steps within a parent achievement.

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Sub-achievement identifier |
| `category` | string | Grouping category |
| `name` | string | Display name |
| `description` | string | Description |
| `count` | int64 | Current progress |
| `max_count` | int64 | Completion threshold |
| `claim_time_sec` | int64 | Claim timestamp |
| `reset_time_sec` | int64 | Reset schedule |
| `expire_time_sec` | int64 | Expiration time |
| `precondition_ids` | []string | Prerequisites (other sub-achievement IDs) |
| `reward` | Reward | Rolled reward |
| `available_rewards` | AvailableRewards | Reward definition |
| `additional_properties` | map[string]string | Custom metadata |
| `auto_claim` | bool | Auto-claim flag |
| `auto_reset` | bool | Auto-reset flag |

### AchievementList (Storage Object)

The per-user storage object contains two maps:

```json
{
  "achievements": {
    "<id>": { /* Achievement */ }
  },
  "repeat_achievements": {
    "<id>": { /* Achievement (repeatable) */ }
  }
}
```

---

## Configuration

### Config Structure

Achievement definitions are provided as JSON config at server initialization.

```json
{
  "achievements": {
    "<achievement_id>": {
      "name": "kill_100_enemies",
      "category": "combat",
      "description": "kill_100_enemies_desc",
      "max_count": 100,
      "count": 0,
      "is_repeatable": false,
      "auto_claim": false,
      "auto_claim_total": false,
      "auto_reset": false,
      "reset_cronexpr": "",
      "duration_sec": 0,
      "start_time_sec": 0,
      "end_time_sec": 0,
      "precondition_ids": [],
      "reward": {
        /* EconomyConfigReward */
      },
      "total_reward": {
        /* EconomyConfigReward — granted when all sub-achievements claimed */
      },
      "additional_properties": {
        "custom_key": "custom_value"
      },
      "sub_achievements": {
        "<sub_id>": {
          "name": "kill_10_enemies",
          "max_count": 10,
          "reward": { /* EconomyConfigReward */ },
          "additional_properties": {}
        }
      }
    }
  }
}
```

### Config Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | required | Display name / i18n key |
| `category` | string | required | Grouping for filtering |
| `description` | string | "" | Description / i18n key |
| `max_count` | int64 | required | Progress target |
| `count` | int64 | 0 | Initial progress (pre-seeded) |
| `is_repeatable` | bool | false | Cycles after claim |
| `auto_claim` | bool | false | Claim automatically on completion |
| `auto_claim_total` | bool | false | Claim total reward automatically |
| `auto_reset` | bool | false | Reset count after claim |
| `reset_cronexpr` | string | "" | CRON expression for periodic reset |
| `duration_sec` | int64 | 0 | Time-limited duration (0 = permanent) |
| `start_time_sec` | int64 | 0 | Activation timestamp |
| `end_time_sec` | int64 | 0 | Deactivation timestamp |
| `precondition_ids` | []string | [] | Achievement IDs that must be claimed first |
| `reward` | EconomyConfigReward | null | Per-claim reward definition |
| `total_reward` | EconomyConfigReward | null | Completion reward (all subs done) |
| `additional_properties` | map | {} | Arbitrary key-value metadata |
| `sub_achievements` | map | {} | Nested sub-achievement definitions |

### Achievement Patterns via Config

**Daily quest:** `reset_cronexpr: "0 0 * * *"`, `auto_reset: true`

**Weekly quest:** `reset_cronexpr: "0 0 * * 1"`, `auto_reset: true`

**Tiered achievement:** Parent with sub-achievements as tiers, `precondition_ids` chaining tiers in order, `total_reward` on parent for completing all tiers.

**Time-limited event:** `start_time_sec` / `end_time_sec` or `duration_sec` set.

**Repeatable grind:** `is_repeatable: true`, `auto_reset: true` — cycles indefinitely.

---

## API

### Interface

```go
type AchievementsSystem interface {
    // Progress tracking
    UpdateAchievements(ctx, logger, nk, userID string,
        updates map[string]int64,
    ) (achievements, repeatAchievements map[string]*Achievement, err error)

    // Reward claiming
    ClaimAchievements(ctx, logger, nk, userID string,
        achievementIDs []string, claimTotal bool,
    ) (achievements, repeatAchievements map[string]*Achievement, err error)

    // State retrieval
    GetAchievements(ctx, logger, nk, userID string,
    ) (achievements, repeatAchievements map[string]*Achievement, err error)

    // Mark claimed without granting rewards (external reward systems)
    MarkAchievementsAsClaimed(ctx, logger, nk, userID string,
        achievementIDs []string,
    ) error

    // Reward hooks
    SetOnAchievementReward(fn OnReward[*AchievementsConfigAchievement])
    AddOnAchievementReward(fn OnReward[*AchievementsConfigAchievement])
    SetOnSubAchievementReward(fn OnReward[*AchievementsConfigSubAchievement])
    AddOnSubAchievementReward(fn OnReward[*AchievementsConfigSubAchievement])
    SetOnAchievementTotalReward(fn OnReward[*AchievementsConfigAchievement])
    AddOnAchievementTotalReward(fn OnReward[*AchievementsConfigAchievement])
}
```

### Reward Hook Signature

```go
type OnReward[T any] func(
    ctx context.Context,
    logger runtime.Logger,
    nk runtime.NakamaModule,
    userID string,
    sourceID string,      // achievement ID
    source T,             // config struct
    rewardConfig *EconomyConfigReward,
    reward *Reward,       // rolled reward (mutable)
) (*Reward, error)
```

Hooks execute in registration order. Each hook receives the reward from the previous hook and can modify it.

### RPC Endpoints

| RPC ID | Name | Request | Response |
|--------|------|---------|----------|
| 16 | `RPC_ID_ACHIEVEMENTS_CLAIM` | `AchievementsClaimRequest` | `AchievementsUpdateAck` |
| 17 | `RPC_ID_ACHIEVEMENTS_GET` | (empty) | `AchievementList` |
| 18 | `RPC_ID_ACHIEVEMENTS_UPDATE` | `AchievementsUpdateRequest` | `AchievementsUpdateAck` |

**Request types:**

```protobuf
message AchievementsClaimRequest {
    repeated string ids = 1;
    bool claim_total_reward = 2;
}

message AchievementsUpdateRequest {
    map<string, int64> achievements = 3;  // achievement_id → delta
}

message AchievementsUpdateAck {
    map<string, Achievement> achievements = 1;
    map<string, Achievement> repeat_achievements = 2;
}
```

Additional JSON-only RPCs exist for admin/tooling:
- `AchievementsList` — filtered listing by category
- `AchievementsProgress` — single achievement update
- `AchievementDetails` — full metadata for one achievement

---

## Storage

### Location

| Property | Value |
|----------|-------|
| Collection | `achievements` |
| Key | `user_achievements` |
| Permission | Owner Read / Owner Write |
| Backend | Nakama Storage → CockroachDB |
| Format | JSON-serialized `AchievementList` |

Achievements do **not** use Nakama's built-in Achievement API. They are stored as a single JSON object per user in Nakama storage for maximum flexibility.

### Concurrency

Storage reads and writes use Nakama's **optimistic locking** (version field). If a concurrent write occurs, the operation retries with the latest state.

---

## Core Algorithms

### UpdateAchievements

```
For each (achievementID, delta) in updates:
  1. Load user's AchievementList from storage
  2. Look up achievement config
  3. If not in user state, create from config defaults
  4. Check time constraints:
     - Skip if expired (duration_sec exceeded)
     - If reset_time passed, reset count and calculate next reset from CRON
  5. Check preconditions (all precondition_ids must be claimed)
  6. Increment count: min(count + delta, max_count)
  7. If sub-achievements exist, propagate update to matching sub-achievements
  8. If auto_claim and count >= max_count:
     → Trigger ClaimAchievements internally
  9. Write updated state to storage
  10. Return updated achievements
```

### ClaimAchievements

```
For each achievementID:
  1. Load user's AchievementList from storage
  2. Validate:
     - Achievement exists in user state
     - count >= max_count (completed)
     - claim_time_sec == 0 (not already claimed) OR is_repeatable
     - Preconditions met
  3. Roll reward: EconomySystem.RewardRoll(rewardConfig)
  4. Execute OnAchievementReward hooks (in order):
     hook(ctx, logger, nk, userID, achievementID, config, rewardConfig, reward) → modified reward
  5. Grant reward: EconomySystem.RewardGrant(userID, reward, metadata)
     metadata = { "achievement_id": id, "type": "standard"|"repeat"|"total" }
  6. Set claim_time_sec = now
  7. If auto_reset:
     - Reset count = 0, claim_time_sec = 0
     - Calculate next reset_time from CRON
  8. If claimTotal and all sub-achievements claimed:
     - Roll and grant total_reward
     - Execute OnAchievementTotalReward hooks
     - Set total_claim_time_sec = now
  9. Write to storage
  10. Return claimed achievements
```

### Repeat Achievement Cycling

When a repeatable achievement is claimed and auto-reset is enabled:

```
count = count % max_count   // carry over excess progress
claim_time_sec = 0          // allow next claim cycle
```

### Precondition Check

```
For each precondition_id:
  achievement = user_state[precondition_id]
  if not found OR claim_time_sec == 0:
    return false
return true
```

### CRON Reset Calculation

```
schedule = cronParser.Parse(reset_cronexpr)
next_reset = schedule.Next(now)
reset_time_sec = next_reset.Unix()
```

---

## Reward Hook Chain

The hook system allows game modules to inject custom behavior into the claim flow without modifying the core achievement system.

```
ClaimAchievements
  │
  ├─ RewardRoll(config) → base reward
  │
  ├─ Hook 1: Quest module
  │   └─ Reads additional_properties["token_reward"]
  │   └─ Grants quest tokens via TokenManager
  │   └─ Returns reward (unmodified or modified)
  │
  ├─ Hook 2: Coop module
  │   └─ Reads additional_properties["coop_point"]
  │   └─ Grants coop points via PointManager
  │   └─ First-claim bonus: grants gems
  │   └─ Returns reward
  │
  ├─ Hook N: (any additional game module)
  │
  └─ RewardGrant(final reward)
```

### Hook Types

| Hook | Fires When | Config Type |
|------|-----------|-------------|
| `OnAchievementReward` | Standard achievement claimed | `AchievementsConfigAchievement` |
| `OnSubAchievementReward` | Sub-achievement claimed | `AchievementsConfigSubAchievement` |
| `OnAchievementTotalReward` | All subs complete, total claimed | `AchievementsConfigAchievement` |

### Registration

- `Set*` — replaces all hooks with one function
- `Add*` — appends to the hook chain (preferred for multi-module setups)

---

## Initialization

### Bootstrap Sequence

```go
pamlogix.Init(ctx, logger, nk, initializer,
    []pamlogix.InitOption{
        pamlogix.WithAchievementStorage(achievementStorage),
    },
    pamlogix.WithAchievementsSystem(configPath, registerRPCs),
    // ... other systems
)
```

| Parameter | Description |
|-----------|-------------|
| `configPath` | Path to merged achievements JSON config |
| `registerRPCs` | If true, registers built-in RPC handlers |
| `WithAchievementStorage` | Optional custom storage adapter |

### Module Hook Registration

After initialization, game modules register their reward hooks:

```go
achievementsSystem.AddOnAchievementReward(questRewardHook)
achievementsSystem.AddOnAchievementReward(coopRewardHook)
achievementsSystem.AddOnSubAchievementReward(subAchievementHook)
```

---

## Integration Patterns

### Pattern 1: Event-Driven Tracking

Map gameplay events to achievement IDs, update progress on each event.

```go
// Event mapping config: event → []achievementID
eventMap := map[string][]string{
    "enemy_kill":      {"d0101", "w0201", "lifetime_kills"},
    "campaign_clear":  {"d0301", "w0301"},
    "gacha_pull":      {"d0401"},
}

// On event
func onEnemyKill(userID string, count int64) {
    updates := map[string]int64{}
    for _, id := range eventMap["enemy_kill"] {
        updates[id] = count
    }
    achievementsSystem.UpdateAchievements(ctx, logger, nk, userID, updates)
}
```

### Pattern 2: Tiered Sub-Achievements

Use sub-achievements with precondition chains for tiered progress.

```json
{
  "combat_master": {
    "name": "Combat Master",
    "category": "combat",
    "max_count": 1,
    "auto_claim_total": true,
    "total_reward": { /* grand reward */ },
    "sub_achievements": {
      "tier_1": { "max_count": 10, "reward": { /* bronze */ } },
      "tier_2": { "max_count": 50, "precondition_ids": ["tier_1"], "reward": { /* silver */ } },
      "tier_3": { "max_count": 200, "precondition_ids": ["tier_2"], "reward": { /* gold */ } }
    }
  }
}
```

### Pattern 3: Daily/Weekly Reset Quests

CRON-based reset with auto-reset for recurring quests.

```json
{
  "daily_kill_10": {
    "name": "Daily: Kill 10 enemies",
    "category": "daily",
    "max_count": 10,
    "auto_reset": true,
    "reset_cronexpr": "0 0 * * *",
    "reward": { /* daily reward */ }
  }
}
```

### Pattern 4: Custom Side Effects via additional_properties

Use `additional_properties` to pass game-specific data to reward hooks without changing the achievement system.

```json
{
  "coop_stage_clear_1": {
    "additional_properties": {
      "coop_point": "5",
      "achievement_type": "1",
      "tier": "1"
    }
  }
}
```

```go
achievementsSystem.AddOnAchievementReward(func(
    ctx, logger, nk, userID, sourceID string,
    source *AchievementsConfigAchievement,
    rewardConfig, reward,
) (*Reward, error) {
    if points, ok := source.AdditionalProperties["coop_point"]; ok {
        pointManager.Grant(userID, parseInt(points))
    }
    return reward, nil
})
```

### Pattern 5: External Reward Management

For systems that manage their own rewards (e.g., chest systems), use `MarkAchievementsAsClaimed` to update state without triggering the reward pipeline.

```go
// Grant custom rewards externally
grantChestReward(userID, chestTier)
// Then mark the achievement as claimed
achievementsSystem.MarkAchievementsAsClaimed(ctx, logger, nk, userID, []string{"chest_tier_5"})
```

---

## Data Flow — Complete Example

```
Player kills an enemy
  │
  ▼
Game module: questTracker.TrackEnemyKill(userID, 1)
  │
  ▼
Event mapping: "enemy_kill" → ["d0101", "w0102", "lifetime_kills"]
  │
  ▼
AchievementsSystem.UpdateAchievements(userID, {"d0101": 1, "w0102": 1, "lifetime_kills": 1})
  │
  ├─ Read user state from Nakama storage
  ├─ d0101: count 9 → 10 (max_count=10, auto_claim=true)
  │   └─ Triggers auto-claim:
  │       ├─ RewardRoll() → 100 gold
  │       ├─ Hook: quest module → grants 1 daily token
  │       ├─ RewardGrant(100 gold)
  │       └─ auto_reset → count=0, next reset from CRON
  ├─ w0102: count 45 → 46 (max_count=50, not complete yet)
  ├─ lifetime_kills: count 9999 → 10000 (max_count=10000)
  │   └─ Complete! Awaits manual claim from client
  ├─ Write updated state to Nakama storage
  │
  ▼
Return: updated achievement states to caller
  │
  ▼
Client receives notification, updates UI
```

---

## Error Handling

| Error | Condition |
|-------|-----------|
| `ErrAchievementNotFound` | Achievement ID not in config |
| `ErrAchievementSystemUnavailable` | System not initialized |
| `ErrAchievementLoadFailed` | Config file parse error |
| `NewAchievementStorageError` | Storage read/write failure |
| `NewAchievementSerializationError` | JSON marshal/unmarshal failure |
| `NewAchievementParseError` | User state corruption |

Storage errors trigger retries via Nakama's optimistic locking. Config errors are fatal at startup.

COpied