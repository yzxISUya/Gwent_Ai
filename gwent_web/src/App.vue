<template>
  <div class="app-shell">
    <!-- Top Control Bar -->
    <header class="top-bar">
      <div class="brand">
        <span class="brand-badge">GWENT · RL</span>
        <span class="brand-sub">对弈与 AI 决策透视平台</span>
      </div>

      <div class="controls-row">
        <label class="ctrl-item">
          <span>模式</span>
          <select v-model="gameMode" @change="onModeChange">
            <option value="human-vs-ai">人机对战 (P0 人类 vs P1 AI)</option>
            <option value="ai-vs-ai">AI 擂台观战 (P0 AI vs P1 AI)</option>
          </select>
        </label>

        <label class="ctrl-item">
          <span>P0 卡组</span>
          <select v-model="preset0" @change="startNewGame()">
            <option v-for="p in presetList" :key="p.id" :value="p.id">
              [{{ factionShort(p.faction) }}] {{ p.nameZh }}
            </option>
          </select>
        </label>

        <label v-if="gameMode === 'ai-vs-ai'" class="ctrl-item">
          <span>P0 引擎</span>
          <select v-model="aiModel0">
            <option v-for="m in aiOptions" :key="m.value" :value="m.value">{{ m.label }}</option>
          </select>
        </label>

        <label class="ctrl-item">
          <span>P1 卡组</span>
          <select v-model="preset1" @change="startNewGame()">
            <option v-for="p in presetList" :key="p.id" :value="p.id">
              [{{ factionShort(p.faction) }}] {{ p.nameZh }}
            </option>
          </select>
        </label>

        <label class="ctrl-item">
          <span>P1 引擎</span>
          <select v-model="aiModel1">
            <option v-for="m in aiOptions" :key="m.value" :value="m.value">{{ m.label }}</option>
          </select>
        </label>

        <label class="ctrl-item seed-ctrl">
          <span>Seed</span>
          <input type="number" v-model.number="seedInput" />
        </label>

        <button class="btn primary" @click="startNewGame(seedInput)">新对局</button>
        <button class="btn" @click="startNewGame(Math.floor(Math.random() * 900000) + 100000)">
          随机种子
        </button>

        <template v-if="gameMode === 'ai-vs-ai'">
          <button class="btn accent-outline" @click="toggleAutoPlay">
            {{ autoPlay ? '暂停观战' : '自动对弈' }}
          </button>
          <button class="btn" :disabled="view.done" @click="stepAiManually">单步推进</button>
        </template>
      </div>
    </header>

    <!-- Main Workspace -->
    <main class="workspace">
      <!-- Left / Center: Minimalist Battlefield -->
      <section class="board-panel">
        <!-- Player 1 (Top / Opponent) Header -->
        <div class="player-bar" :class="{ 'active-turn': !view.done && view.turn === 1 }">
          <div class="p-ident">
            <span class="p-tag">P1 {{ gameMode === 'human-vs-ai' ? 'AI' : 'AI-1' }}</span>
            <span class="p-faction">{{ factionName(view.players[1].faction) }}</span>
            <span class="leader-pill" :class="{ spent: !view.players[1].leaderAvailable }">
              领袖: {{ leaderName(view.players[1].leader_id) }}
              [{{ view.players[1].leaderAvailable ? '可用' : '已用/被动' }}]
            </span>
          </div>
          <div class="p-stats">
            <span class="gems">
              胜局宝石
              <span v-for="g in 2" :key="g" class="gem-dot" :class="{ on: g <= view.gems[1] }"></span>
            </span>
            <span class="stat-item">手牌<b>{{ view.players[1].hand.length }}</b></span>
            <span class="stat-item">牌库<b>{{ view.players[1].deck.length }}</b></span>
            <span class="stat-item">墓地<b>{{ view.players[1].graveyard.length }}</b></span>
            <span v-if="view.players[1].passed" class="passed-badge">已放弃 (PASSED)</span>
            <span class="total-score">{{ view.scores[1] }}</span>
          </div>
        </div>

        <!-- Player 1 Rows: Siege, Ranged, Close -->
        <div class="rows-group opponent-rows">
          <div
            v-for="rKey in ['siege', 'ranged', 'close']"
            :key="'p1-' + rKey"
            class="battle-row"
            :class="{ weathered: isRowWeathered(rKey) }"
          >
            <div class="row-meta">
              <span class="row-title">{{ rowLabel(rKey) }}</span>
              <span class="row-badges">
                <span v-if="isRowWeathered(rKey)" class="badge">{{ weatherBadge(rKey) }}</span>
                <span v-if="view.players[1].rowFlags[rKey].hornActive" class="badge">号角×2</span>
              </span>
              <span class="row-score">{{ view.rowScores[1][rKey] }}</span>
            </div>
            <div class="row-cards">
              <div
                v-for="u in view.players[1].computedRows[rKey]"
                :key="u.uid"
                class="unit-chip"
                :class="{
                  hero: u.card._is_hero,
                  buffed: u.power > u.card.strength,
                  debuffed: u.power < u.card.strength,
                }"
                :title="u.card.nameZh + ' | ' + (u.card.abilityZh || '普通单位')"
              >
                <span class="u-pwr">{{ u.power }}</span>
                <span class="u-name">{{ u.card.nameZh }}</span>
                <span v-if="abilityShort(u.card)" class="u-ab">{{ abilityShort(u.card) }}</span>
              </div>
            </div>
          </div>
        </div>

        <!-- Center Status & Weather Strip -->
        <div class="center-strip">
          <div class="round-info">
            <span class="round-pill">第 {{ view.round }} 小局</span>
            <span class="phase-pill">
              {{ subPhaseText }}
            </span>
          </div>
          <div class="weather-strip">
            <span class="w-label">全局天气</span>
            <span class="w-tag" :class="{ active: view.weather.frost }">刺骨冰霜 (近战=1)</span>
            <span class="w-tag" :class="{ active: view.weather.fog }">浓雾 (远程=1)</span>
            <span class="w-tag" :class="{ active: view.weather.rain }">暴雨 (攻城=1)</span>
          </div>
          <div v-if="view.done" class="gameover-banner">
            对局结算：{{
              view.winner === null
                ? '平局'
                : view.winner === 0
                  ? 'P0 获胜'
                  : 'P1 获胜'
            }}
          </div>
        </div>

        <!-- Player 0 Rows: Close, Ranged, Siege -->
        <div class="rows-group player-rows">
          <div
            v-for="rKey in ['close', 'ranged', 'siege']"
            :key="'p0-' + rKey"
            class="battle-row"
            :class="{ weathered: isRowWeathered(rKey) }"
          >
            <div class="row-meta">
              <span class="row-title">{{ rowLabel(rKey) }}</span>
              <span class="row-badges">
                <span v-if="isRowWeathered(rKey)" class="badge">{{ weatherBadge(rKey) }}</span>
                <span v-if="view.players[0].rowFlags[rKey].hornActive" class="badge">号角×2</span>
              </span>
              <span class="row-score">{{ view.rowScores[0][rKey] }}</span>
            </div>
            <div class="row-cards">
              <div
                v-for="u in view.players[0].computedRows[rKey]"
                :key="u.uid"
                class="unit-chip"
                :class="{
                  hero: u.card._is_hero,
                  buffed: u.power > u.card.strength,
                  debuffed: u.power < u.card.strength,
                  selectable: canSelectDecoyTarget(u.card.idx),
                }"
                @click="onClickBoardUnit(u.card.idx)"
              >
                <span class="u-pwr">{{ u.power }}</span>
                <span class="u-name">{{ u.card.nameZh }}</span>
                <span v-if="abilityShort(u.card)" class="u-ab">{{ abilityShort(u.card) }}</span>
              </div>
            </div>
          </div>
        </div>

        <!-- Player 0 (Bottom / Human or AI-0) Header -->
        <div class="player-bar" :class="{ 'active-turn': !view.done && view.turn === 0 }">
          <div class="p-ident">
            <span class="p-tag">P0 {{ gameMode === 'human-vs-ai' ? '玩家' : 'AI-0' }}</span>
            <span class="p-faction">{{ factionName(view.players[0].faction) }}</span>
            <span class="leader-pill" :class="{ spent: !view.players[0].leaderAvailable }">
              领袖: {{ leaderName(view.players[0].leader_id) }}
              [{{ view.players[0].leaderAvailable ? '可用' : '已用/被动' }}]
            </span>
          </div>
          <div class="p-stats">
            <span class="gems">
              胜局宝石
              <span v-for="g in 2" :key="g" class="gem-dot" :class="{ on: g <= view.gems[0] }"></span>
            </span>
            <span class="stat-item">手牌<b>{{ view.players[0].hand.length }}</b></span>
            <span class="stat-item">牌库<b>{{ view.players[0].deck.length }}</b></span>
            <span class="stat-item">墓地<b>{{ view.players[0].graveyard.length }}</b></span>
            <span v-if="view.players[0].passed" class="passed-badge">已放弃 (PASSED)</span>
            <span class="total-score">{{ view.scores[0] }}</span>
          </div>
        </div>

        <!-- Interactive Action & Hand Zone -->
        <div class="hand-dock">
          <div class="dock-header">
            <span class="dock-title">
              <template v-if="view.subPhase === 'MEDIC'">
                [医师复活] 请从下方墓地列表选择 1 张非英雄单位立即打出
              </template>
              <template v-else-if="view.subPhase === 'DECOY'">
                [诱饵回收] 请点击己方场上 1 张非英雄单位将其换回手牌
              </template>
              <template v-else-if="view.subPhase === 'MULLIGAN'">
                [开局调度] P{{ view.turn }} 剩余可换 {{ view.players[view.turn].mulligan_left }} 张牌（点击卡牌更换，或点击右侧结束换牌）
              </template>
              <template v-else>
                手牌与战术动作（点击卡牌直接打出）
              </template>
            </span>

            <div class="special-actions" v-if="isHumanTurn">
              <button
                v-for="(rKey, rIdx) in ['close', 'ranged', 'siege']"
                :key="'horn-' + rKey"
                v-show="view.legalMask[hornRowOffset + rIdx]"
                class="btn small accent-outline"
                @click="commitHumanAction(hornRowOffset + rIdx)"
              >
                指挥官号角 → {{ rowLabel(rKey) }}
              </button>
              <button
                v-if="view.legalMask[useLeaderAction]"
                class="btn small accent-outline"
                @click="commitHumanAction(useLeaderAction)"
              >
                发动领袖技能
              </button>
              <button
                v-if="view.legalMask[passAction]"
                class="btn small"
                @click="commitHumanAction(passAction)"
              >
                {{ view.subPhase === 'MULLIGAN' ? '结束换牌' : '放弃本小局 (Pass)' }}
              </button>
            </div>
          </div>

          <!-- Medic Graveyard Picker -->
          <div v-if="view.subPhase === 'MEDIC'" class="hand-cards">
            <button
              v-for="item in medicCandidates"
              :key="item.uid"
              class="card-tile"
              :disabled="!isHumanTurn"
              @click="commitHumanAction(item.card.idx)"
            >
              <div class="tile-top">
                <span class="tile-str">{{ item.card.strength }}</span>
                <span class="tile-row">{{ rowShort(item.card._primary_row) }}</span>
              </div>
              <div class="tile-name">{{ item.card.nameZh }}</div>
              <div class="tile-ab">{{ abilityShort(item.card) || '普通单位' }}</div>
            </button>
          </div>

          <!-- Normal Hand Cards -->
          <div v-else class="hand-cards">
            <button
              v-for="inst in displayedHand"
              :key="inst.uid"
              class="card-tile"
              :class="{
                hero: inst.card._is_hero,
                special: inst.card.type !== 'unit',
                playable: isHumanTurn && isHandCardPlayable(inst.card),
                suggested: xray.bestAction === inst.card.idx,
              }"
              :disabled="!isHumanTurn || !isHandCardPlayable(inst.card)"
              @click="onClickHandCard(inst.card)"
            >
              <div class="tile-top">
                <span class="tile-str">
                  {{ inst.card.type === 'unit' ? inst.card.strength : '特' }}
                </span>
                <span class="tile-row">{{ rowShort(inst.card._primary_row) || '特殊' }}</span>
              </div>
              <div class="tile-name">{{ inst.card.nameZh }}</div>
              <div class="tile-ab">{{ abilityShort(inst.card) || (inst.card._is_hero ? '英雄' : '普通') }}</div>
            </button>
          </div>
        </div>
      </section>

      <!-- Right Sidebar: AI X-Ray Panel & Match Log -->
      <aside class="xray-panel">
        <div class="xray-card">
          <div class="xray-header">
            <h3>AI 思考透视面板 (X-Ray)</h3>
            <span class="engine-tag">{{ xray.engineUsed || '就绪' }}</span>
          </div>

          <!-- Win-Rate Evaluation Bar -->
          <div class="eval-section">
            <div class="eval-labels">
              <span class="p0-lbl">P0 胜率 {{ xray.p0WinRate.toFixed(1) }}%</span>
              <span class="val-lbl">V(s) = {{ xray.value >= 0 ? '+' : '' }}{{ xray.value.toFixed(3) }}</span>
              <span class="p1-lbl">P1 胜率 {{ (100 - xray.p0WinRate).toFixed(1) }}%</span>
            </div>
            <div class="eval-bar-track">
              <div class="eval-bar-fill" :style="{ width: xray.p0WinRate + '%' }"></div>
            </div>
          </div>

          <!-- Win-Rate History Sparkline -->
          <div class="sparkline-box" v-if="winRateHistory.length > 1">
            <div class="spark-caption">对局实时胜率走势 (P0 视角 0%~100%)</div>
            <svg viewBox="0 0 260 60" class="spark-svg">
              <line x1="0" y1="30" x2="260" y2="30" stroke="#e5e5e5" stroke-width="1" />
              <polyline
                fill="none"
                stroke=var(--accent-70)
                stroke-width="1.75"
                :points="sparklinePoints"
              />
            </svg>
          </div>

          <!-- Top-K Candidate Actions Distribution -->
          <div class="topk-section">
            <div class="topk-title">
              <span>候选动作概率 Top-K (P{{ view.turn }})</span>
              <button
                v-if="isHumanTurn && xray.bestAction !== null"
                class="btn small accent-outline"
                @click="commitHumanAction(xray.bestAction)"
              >
                采纳 AI 首选
              </button>
            </div>
            <div v-if="xray.topActions.length === 0" class="empty-hint">对局已结束</div>
            <div v-for="(item, idx) in xray.topActions" :key="item.action" class="topk-row">
              <div class="topk-meta">
                <span class="topk-rank">#{{ idx + 1 }} {{ item.label }}</span>
                <span class="topk-prob">{{ (item.prob * 100).toFixed(1) }}%</span>
              </div>
              <div class="topk-bar-bg">
                <div class="topk-bar-fg" :style="{ width: Math.min(100, item.prob * 100) + '%' }"></div>
              </div>
            </div>
          </div>
        </div>

        <!-- Round History & Action Log -->
        <div class="log-card">
          <h3>小局比分与动作记录 (Step {{ actionHistory.length }})</h3>
          <div v-if="view.roundHistory.length" class="round-hist-list">
            <div v-for="rh in view.roundHistory" :key="rh.round" class="rh-item">
              第 {{ rh.round }} 局：P0 <b>{{ rh.scores[0] }}</b> : <b>{{ rh.scores[1] }}</b> P1
              — {{ rh.winner === null ? '平局' : `P${rh.winner} 胜` }}
            </div>
          </div>
          <div class="log-list">
            <div v-for="(entry, i) in matchLogs.slice().reverse()" :key="i" class="log-line">
              {{ entry }}
            </div>
          </div>
        </div>
      </aside>
    </main>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted, onBeforeUnmount } from 'vue';
import {
  GwentEngine,
  HeuristicAI,
  computeRowPowers,
  CARDS_BY_IDX,
  LEADERS_BY_ID,
  FACTIONS_BY_ID,
  PRESET_LIST,
  ROW_KEYS,
  WEATHER_FOR_ROW,
  HORN_ROW_OFFSET,
  USE_LEADER_ACTION,
  PASS_ACTION,
} from './engine/engine.js';

const presetList = PRESET_LIST;
const hornRowOffset = HORN_ROW_OFFSET;
const useLeaderAction = USE_LEADER_ACTION;
const passAction = PASS_ACTION;

const aiOptions = [
  { value: 'onnx:ppo-v1', label: 'RL · Masked PPO (ONNX)', url: '/models/ppo_v1.onnx' },
  { value: 'onnx:dmc-v1', label: 'RL · DouZero DMC (ONNX)', url: '/models/dmc_v1.onnx' },
  { value: 'ismcts', label: 'ISMCTS · 信息集蒙特卡洛搜索', url: null },
  { value: 'heuristic-hard', label: '启发式 AI · 困难 (1-ply 推演)', url: null },
  { value: 'heuristic-normal', label: '启发式 AI · 普通', url: null },
  { value: 'heuristic-easy', label: '启发式 AI · 简单', url: null },
];

const gameMode = ref('human-vs-ai');
const preset0 = ref('nr-temeria');
const preset1 = ref('ng-white-flame');
const aiModel0 = ref('onnx:ppo-v1');
const aiModel1 = ref('onnx:ppo-v1');
const seedInput = ref(20261106);
const autoPlay = ref(false);

let engine = new GwentEngine(seedInput.value, preset0.value, preset1.value);
const actionHistory = ref([]);
const matchLogs = ref([]);
const winRateHistory = ref([50.0]);

const view = reactive({
  round: 1,
  turn: 0,
  subPhase: 'MULLIGAN',
  gems: [2, 2],
  scores: [0, 0],
  rowScores: [
    { close: 0, ranged: 0, siege: 0 },
    { close: 0, ranged: 0, siege: 0 },
  ],
  weather: { frost: false, fog: false, rain: false },
  players: [
    {
      faction: 'northern-realms',
      leader_id: '',
      leaderAvailable: false,
      hand: [],
      deck: [],
      graveyard: [],
      mulligan_left: 2,
      passed: false,
      computedRows: { close: [], ranged: [], siege: [] },
      rowFlags: { close: {}, ranged: {}, siege: {} },
    },
    {
      faction: 'nilfgaard',
      leader_id: '',
      leaderAvailable: false,
      hand: [],
      deck: [],
      graveyard: [],
      mulligan_left: 2,
      passed: false,
      computedRows: { close: [], ranged: [], siege: [] },
      rowFlags: { close: {}, ranged: {}, siege: {} },
    },
  ],
  legalMask: new Array(91).fill(false),
  roundHistory: [],
  done: false,
  winner: null,
});

const xray = reactive({
  bestAction: null,
  bestLabel: '',
  value: 0.0,
  p0WinRate: 50.0,
  topActions: [],
  engineUsed: 'onnx:ppo-v1',
});

let worker = null;
let reqSeq = 0;
let autoTimer = null;

function syncView() {
  view.round = engine.round;
  view.turn = engine.turn;
  view.subPhase = engine.sub_phase;
  view.gems = [...engine.gems];
  view.scores = [...engine.scores];
  view.rowScores = [{ ...engine.row_scores[0] }, { ...engine.row_scores[1] }];
  view.weather = { ...engine.weather };
  view.legalMask = engine.getLegalMask();
  view.roundHistory = engine.round_history.map((h) => ({ ...h }));
  view.done = engine.done;
  view.winner = engine.winner;

  for (const pIdx of [0, 1]) {
    const pl = engine.players[pIdx];
    const target = view.players[pIdx];
    target.faction = pl.faction;
    target.leader_id = pl.leader_id;
    target.leaderAvailable = engine.isLeaderAvailable(pIdx);
    target.hand = pl.hand.map((inst) => ({ ...inst, card: CARDS_BY_IDX[inst.card_idx] }));
    target.deck = pl.deck.slice();
    target.graveyard = pl.graveyard.map((inst) => ({ ...inst, card: CARDS_BY_IDX[inst.card_idx] }));
    target.mulligan_left = pl.mulligan_left;
    target.passed = pl.passed;

    for (const r of ROW_KEYS) {
      const hasHorn = pl.row_horn[r] !== null;
      const wActive = engine.weather[WEATHER_FOR_ROW[r]];
      const { powers, hornActive } = computeRowPowers(pl.rows[r], hasHorn, wActive, engine.double_spy);
      target.rowFlags[r] = { hornActive, wActive };
      target.computedRows[r] = pl.rows[r].map((inst, idx) => ({
        ...inst,
        card: CARDS_BY_IDX[inst.card_idx],
        power: powers[idx],
      }));
    }
  }
}

const isHumanTurn = computed(() => {
  if (view.done) return false;
  return gameMode.value === 'human-vs-ai' && view.turn === 0;
});

const displayedHand = computed(() => {
  const pIdx = gameMode.value === 'ai-vs-ai' ? view.turn : 0;
  return view.players[pIdx].hand;
});

const medicCandidates = computed(() => {
  if (view.subPhase !== 'MEDIC') return [];
  const seen = new Set();
  const out = [];
  for (const inst of view.players[view.turn].graveyard) {
    if (inst.card._is_unit && !inst.card._is_hero && !seen.has(inst.card.idx)) {
      seen.add(inst.card.idx);
      out.push(inst);
    }
  }
  return out;
});

const subPhaseText = computed(() => {
  if (view.done) return '对局结束';
  const map = {
    MULLIGAN: `P${view.turn} 开局换牌`,
    PLAY: `P${view.turn} 出牌回合`,
    MEDIC: `P${view.turn} 医师复活选牌`,
    DECOY: `P${view.turn} 诱饵回收选牌`,
  };
  return map[view.subPhase] || view.subPhase;
});

const sparklinePoints = computed(() => {
  const arr = winRateHistory.value;
  if (arr.length < 2) return '0,30 260,30';
  return arr
    .map((wr, idx) => {
      const x = (idx / (arr.length - 1)) * 260;
      const y = 56 - (wr / 100) * 52;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');
});

function isHandCardPlayable(card) {
  if (view.subPhase === 'MULLIGAN') return view.legalMask[card.idx];
  if (view.subPhase !== 'PLAY') return false;
  if (card.id === 'commander-s-horn') {
    return (
      view.legalMask[HORN_ROW_OFFSET] ||
      view.legalMask[HORN_ROW_OFFSET + 1] ||
      view.legalMask[HORN_ROW_OFFSET + 2]
    );
  }
  return view.legalMask[card.idx];
}

function canSelectDecoyTarget(cardIdx) {
  return isHumanTurn.value && view.subPhase === 'DECOY' && view.legalMask[cardIdx];
}

function onClickBoardUnit(cardIdx) {
  if (canSelectDecoyTarget(cardIdx)) {
    commitHumanAction(cardIdx);
  }
}

function onClickHandCard(card) {
  if (!isHumanTurn.value) return;
  if (view.subPhase === 'MULLIGAN') {
    commitHumanAction(card.idx);
    return;
  }
  if (card.id === 'commander-s-horn') {
    for (let i = 0; i < 3; i++) {
      if (view.legalMask[HORN_ROW_OFFSET + i]) {
        commitHumanAction(HORN_ROW_OFFSET + i);
        return;
      }
    }
  } else if (view.legalMask[card.idx]) {
    commitHumanAction(card.idx);
  }
}

function applyStep(action, actorTag = '') {
  if (engine.done) return;
  const p = engine.turn;
  const label = engine.actionLabel(action, 'zh');
  matchLogs.value.push(`[R${engine.round} · P${p}${actorTag}] ${label}`);
  engine.step(action);
  actionHistory.value.push(action);
  syncView();
  requestXRayAndAi();
}

function commitHumanAction(action) {
  if (!isHumanTurn.value) return;
  applyStep(action, ' 玩家');
}

function getModelSpecForPlayer(pIdx) {
  const val = pIdx === 0 ? aiModel0.value : aiModel1.value;
  const found = aiOptions.find((o) => o.value === val) || aiOptions[2];
  return found;
}

function requestXRayAndAi() {
  if (!worker || engine.done) {
    if (engine.done) {
      xray.topActions = [];
      xray.bestAction = null;
      xray.p0WinRate = engine.winner === 0 ? 100 : engine.winner === 1 ? 0 : 50;
      winRateHistory.value.push(xray.p0WinRate);
    }
    return;
  }
  const spec = getModelSpecForPlayer(engine.turn);
  const reqId = ++reqSeq;
  worker.postMessage({
    reqId,
    modelType: spec.value,
    modelUrl: spec.url,
    statePayload: {
      seed: seedInput.value,
      preset0: preset0.value,
      preset1: preset1.value,
      actionHistory: [...actionHistory.value],
    },
  });
}

function stepAiManually() {
  if (engine.done || xray.bestAction === null) return;
  applyStep(xray.bestAction, ' AI');
}

function toggleAutoPlay() {
  autoPlay.value = !autoPlay.value;
  if (autoPlay.value && gameMode.value === 'ai-vs-ai' && !engine.done) {
    requestXRayAndAi();
  }
}

function onModeChange() {
  autoPlay.value = false;
  requestXRayAndAi();
}

function startNewGame(newSeed = null) {
  if (newSeed !== null) seedInput.value = newSeed;
  if (autoTimer) clearTimeout(autoTimer);
  engine.reset(seedInput.value, preset0.value, preset1.value);
  actionHistory.value = [];
  matchLogs.value = [`—— 对局开始 (Seed: ${seedInput.value}) ——`];
  winRateHistory.value = [50.0];
  syncView();
  requestXRayAndAi();
}

// Helpers for display
function factionShort(fid) {
  return FACTIONS_BY_ID[fid]?.shortZh || fid;
}
function factionName(fid) {
  return FACTIONS_BY_ID[fid]?.nameZh || fid;
}
function leaderName(lid) {
  return LEADERS_BY_ID[lid]?.nameZh || lid;
}
function rowLabel(rKey) {
  return { close: '近战行', ranged: '远程行', siege: '攻城行' }[rKey] || rKey;
}
function rowShort(rKey) {
  return { close: '近战', ranged: '远程', siege: '攻城' }[rKey] || '';
}
function isRowWeathered(rKey) {
  return view.weather[WEATHER_FOR_ROW[rKey]];
}
function weatherBadge(rKey) {
  return { close: '霜', ranged: '雾', siege: '雨' }[rKey];
}
function abilityShort(card) {
  const tags = [];
  if (card._is_hero) tags.push('英雄');
  if (card._is_spy) tags.push('间谍');
  if (card._is_medic) tags.push('医师');
  if (card._is_bond) tags.push('紧缚');
  if (card._is_morale) tags.push('士气');
  if (card._is_muster) tags.push('集结');
  if (card._is_horn_unit) tags.push('号角');
  return tags.join('·');
}

onMounted(() => {
  worker = new Worker(new URL('./workers/ai.worker.js', import.meta.url), { type: 'module' });
  worker.onmessage = (e) => {
    const data = e.data;
    if (data.reqId !== reqSeq || data.done) return;
    if (data.error) {
      console.error('[AI Worker Error]', data.error);
      matchLogs.value.push(`[系统提示] AI 决策遇到异常: ${data.error}，正在尝试本地启发式兜底...`);
      try {
        const fallbackBot = new HeuristicAI('hard', engine.rng.state ^ 0x999);
        const fallbackAction = fallbackBot.selectAction(engine);
        xray.bestAction = fallbackAction;
        xray.bestLabel = engine.actionLabel(fallbackAction, 'zh');
        xray.engineUsed = '启发式 AI (主线程兜底)';
        xray.topActions = [{ action: fallbackAction, label: xray.bestLabel, prob: 1.0 }];

        const shouldAiAct =
          (gameMode.value === 'human-vs-ai' && view.turn === 1) ||
          (gameMode.value === 'ai-vs-ai' && autoPlay.value);

        if (shouldAiAct && !view.done) {
          autoTimer = setTimeout(() => {
            if (!view.done && data.reqId === reqSeq) {
              applyStep(fallbackAction, ' AI(兜底)');
            }
          }, 240);
        }
      } catch (fbErr) {
        console.error('[Fallback Fatal]', fbErr);
        matchLogs.value.push(`[系统提示] 启发式兜底亦失败: ${fbErr?.message || fbErr}`);
      }
      return;
    }
    xray.bestAction = data.bestAction;
    xray.bestLabel = data.bestLabel;
    xray.value = data.value;
    xray.p0WinRate = data.p0WinRate;
    xray.topActions = data.topActions || [];
    xray.engineUsed = data.engineUsed;
    winRateHistory.value.push(data.p0WinRate);

    const shouldAiAct =
      (gameMode.value === 'human-vs-ai' && view.turn === 1) ||
      (gameMode.value === 'ai-vs-ai' && autoPlay.value);

    if (shouldAiAct && !view.done) {
      autoTimer = setTimeout(() => {
        if (!view.done && data.reqId === reqSeq) {
          applyStep(data.bestAction, ' AI');
        }
      }, 240);
    }
  };
  worker.onerror = (err) => {
    console.error('[AI Worker Fatal Error]', err);
    matchLogs.value.push(`[系统提示] AI Worker 线程异常: ${err?.message || 'Worker Error'}`);
  };
  startNewGame(seedInput.value);
});

onBeforeUnmount(() => {
  if (autoTimer) clearTimeout(autoTimer);
  if (worker) worker.terminate();
});
</script>
