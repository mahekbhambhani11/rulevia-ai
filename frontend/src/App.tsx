import { useEffect, useRef, useState, type MouseEvent } from "react";
import "./App.css";

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ||
  (import.meta.env.PROD ? "" : "http://127.0.0.1:8000")
).replace(/\/+$/, "");

type PageId =
  | "overview"
  | "live"
  | "rules"
  | "videos"
  | "analysis"
  | "backtesting"
  | "review"
  | "audit"
  | "settings";

type NavigationItem = {
  id: PageId;
  label: string;
  icon: string;
};

const navigation: NavigationItem[] = [
  { id: "overview", label: "Overview", icon: "▦" },
  { id: "live", label: "Live Monitoring", icon: "◉" },
  { id: "rules", label: "Rule Builder", icon: "⌘" },
  { id: "videos", label: "Video Library", icon: "▣" },
  { id: "analysis", label: "Analysis Results", icon: "⌁" },
  { id: "backtesting", label: "Backtesting", icon: "↻" },
  { id: "review", label: "Review Queue", icon: "☑" },
  { id: "audit", label: "Audit History", icon: "≡" },
  { id: "settings", label: "Settings", icon: "⚙" },
];

const pageInfo: Record<PageId, { title: string; subtitle: string }> = {
  overview: {
    title: "Safety overview",
    subtitle: "Your industrial safety operations at a glance.",
  },
  live: {
    title: "Live monitoring",
    subtitle: "Monitor camera feeds and detect potential safety risks.",
  },
  rules: {
    title: "Natural-language rule builder",
    subtitle: "Describe a safety requirement in plain English.",
  },
  videos: {
    title: "Video library",
    subtitle: "Manage recorded footage for safety analysis.",
  },
  analysis: {
    title: "Analysis results",
    subtitle: "Review detections and evidence from video processing.",
  },
  backtesting: {
    title: "Historical backtesting",
    subtitle: "Evaluate safety rules against previously recorded footage.",
  },
  review: {
    title: "Review queue",
    subtitle: "Inspect potential safety events before confirming them.",
  },
  audit: {
    title: "Audit history",
    subtitle: "Track rule changes, reviews, and system activity.",
  },
  settings: {
    title: "Settings",
    subtitle: "Configure your workspace and application preferences.",
  },
};

function App() {
  const [activePage, setActivePage] = useState<PageId>("overview");
  const [backendStatus, setBackendStatus] = useState<
    "checking" | "connected" | "offline"
  >("checking");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [selectedSafetyRule, setSelectedSafetyRule] =
    useState<SafetyRuleDraft | null>(null);
  const [videoResult, setVideoResult] = useState<VideoUploadResult | null>(null);
  const [configuredZone, setConfiguredZone] = useState<ZonePoint[]>([]);

  useEffect(() => {
    let active = true;

    async function checkBackend() {
      try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (!response.ok) throw new Error("Health check failed");

        if (active) setBackendStatus("connected");
      } catch {
        if (active) setBackendStatus("offline");
      }
    }

    checkBackend();
    const interval = window.setInterval(checkBackend, 15000);

    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, []);

  const currentPage = pageInfo[activePage];

  function navigateTo(page: PageId) {
    setActivePage(page);
    setSidebarOpen(false);
  }

  return (
    <div className="app-shell">
      {sidebarOpen && (
        <button
          className="mobile-overlay"
          aria-label="Close navigation"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <aside className={`sidebar ${sidebarOpen ? "sidebar-open" : ""}`}>
        <div className="brand">
          <div className="brand-mark">R</div>
          <div className="brand-copy">
            <span className="brand-name">
              rulevia<span>AI</span>
            </span>
            <span className="brand-caption">SAFETY INTELLIGENCE</span>
          </div>
          <button
            className="sidebar-close"
            aria-label="Close navigation"
            onClick={() => setSidebarOpen(false)}
          >
            ×
          </button>
        </div>

        <div className="workspace-label">WORKSPACE</div>

        <nav className="navigation" aria-label="Main navigation">
          {navigation.map((item) => (
            <button
              key={item.id}
              className={`nav-item ${
                activePage === item.id ? "nav-item-active" : ""
              }`}
              onClick={() => navigateTo(item.id)}
              aria-current={activePage === item.id ? "page" : undefined}
            >
              <span className="nav-icon" aria-hidden="true">
                {item.icon}
              </span>
              <span>{item.label}</span>
              {item.id === "review" && <span className="nav-count">0</span>}
            </button>
          ))}
        </nav>

        <div className="sidebar-spacer" />

        <div className="sidebar-help">
          <div className="help-icon">?</div>
          <div>
            <strong>Need help?</strong>
            <p>Build safer workflows with Rulevia.</p>
          </div>
        </div>

        <div className="profile">
          <div className="profile-avatar">R</div>
          <div className="profile-copy">
            <strong>Workspace Admin</strong>
            <span>Local development</span>
          </div>
          <span className="profile-menu">···</span>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="topbar-left">
            <button
              className="menu-toggle"
              aria-label="Open navigation"
              onClick={() => setSidebarOpen(true)}
            >
              ☰
            </button>
            <div className="breadcrumb">
              <span>Workspace</span>
              <span className="breadcrumb-divider">/</span>
              <strong>{currentPage.title}</strong>
            </div>
          </div>

          <div className="topbar-right">
            <div className={`connection-status status-${backendStatus}`}>
              <span className="status-dot" />
              {backendStatus === "checking"
                ? "Checking backend"
                : backendStatus === "connected"
                  ? "Backend connected"
                  : "Backend offline"}
            </div>
            <div className="topbar-avatar">R</div>
          </div>
        </header>

        <div className="page-content">
          <div className="page-heading">
            <div>
              <div className="eyebrow">RULEVIA AI / WORKSPACE</div>
              <h1>{currentPage.title}</h1>
              <p>{currentPage.subtitle}</p>
            </div>

            {activePage === "overview" && (
              <button
                className="primary-button"
                onClick={() => navigateTo("rules")}
              >
                <span>＋</span> Create safety rule
              </button>
            )}

            {activePage === "rules" && (
              <button
                className="primary-button"
                onClick={() => navigateTo("videos")}
              >
                Upload a video <span>→</span>
              </button>
            )}
          </div>

          {activePage === "overview" ? (
            <Overview backendStatus={backendStatus} onNavigate={navigateTo} />
          ) : (
            <FeaturePage
              page={activePage}
              onNavigate={navigateTo}
              selectedSafetyRule={selectedSafetyRule}
              onSelectSafetyRule={setSelectedSafetyRule}
              videoResult={videoResult}
              onVideoResultChange={setVideoResult}
              configuredZone={configuredZone}
              onConfiguredZoneChange={setConfiguredZone}
            />
          )}

          <footer className="page-footer">
            <span>Rulevia AI · Industrial Safety Intelligence</span>
            <span>Development workspace · v1.0.0</span>
          </footer>
        </div>
      </main>
    </div>
  );
}

function Overview({
  backendStatus,
  onNavigate,
}: {
  backendStatus: "checking" | "connected" | "offline";
  onNavigate: (page: PageId) => void;
}) {
  return (
    <>
      <section className="welcome-banner">
        <div className="welcome-content">
          <div className="welcome-tag">
            <span className="welcome-tag-dot" />
            YOUR SAFETY WORKSPACE
          </div>
          <h2>
            Turn safety instructions into
            <br />
            intelligent monitoring.
          </h2>
          <p>
            Define safety rules in plain English, analyze recorded footage, and
            review potential risks with visual evidence.
          </p>
          <button
            className="welcome-button"
            onClick={() => onNavigate("rules")}
          >
            Build your first rule <span>→</span>
          </button>
        </div>
        <div className="welcome-art" aria-hidden="true">
          <div className="art-ring ring-one" />
          <div className="art-ring ring-two" />
          <div className="art-center">
            <span>R</span>
            <small>AI</small>
          </div>
          <div className="art-chip chip-one">RULE ENGINE</div>
          <div className="art-chip chip-two">VIDEO AI</div>
          <div className="art-chip chip-three">EVIDENCE</div>
        </div>
      </section>

      <div className="section-heading">
        <div>
          <h2>Workspace overview</h2>
          <p>Key activity across your safety operations</p>
        </div>
        <span className="demo-label">DEMO DATA</span>
      </div>

      <section className="stats-grid">
        <StatCard
          label="Active safety rules"
          value="—"
          note="No rules configured yet"
          icon="⌘"
          tone="violet"
        />
        <StatCard
          label="Videos analyzed"
          value="—"
          note="Waiting for your first upload"
          icon="▣"
          tone="blue"
        />
        <StatCard
          label="Events for review"
          value="0"
          note="No events generated yet"
          icon="◎"
          tone="amber"
        />
        <StatCard
          label="Backend status"
          value={
            backendStatus === "connected"
              ? "Online"
              : backendStatus === "offline"
                ? "Offline"
                : "…"
          }
          note={
            backendStatus === "connected"
              ? "API health check passed"
              : "Checking local API"
          }
          icon="↗"
          tone="green"
        />
      </section>

      <section className="dashboard-grid">
        <div className="panel monitoring-panel">
          <div className="panel-heading">
            <div>
              <h3>Monitoring overview</h3>
              <p>Camera and video processing status</p>
            </div>
            <span className="neutral-pill">Not configured</span>
          </div>

          <div className="empty-monitor">
            <div className="camera-illustration">
              <svg viewBox="0 0 100 78" fill="none" aria-hidden="true">
                <rect
                  x="12"
                  y="15"
                  width="53"
                  height="38"
                  rx="7"
                  stroke="currentColor"
                  strokeWidth="2.5"
                />
                <path
                  d="M65 27 L87 17 V51 L65 41 Z"
                  stroke="currentColor"
                  strokeWidth="2.5"
                  strokeLinejoin="round"
                />
                <path
                  d="M38 53 V64 M25 65 H51"
                  stroke="currentColor"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                />
                <circle
                  cx="38"
                  cy="34"
                  r="8"
                  stroke="currentColor"
                  strokeWidth="2.5"
                />
              </svg>
            </div>
            <h4>No cameras connected</h4>
            <p>
              Start with recorded footage. Live camera integration will be added
              in a later implementation step.
            </p>
            <button
              className="secondary-button"
              onClick={() => onNavigate("videos")}
            >
              Open video library <span>→</span>
            </button>
          </div>
        </div>

        <div className="panel quick-actions-panel">
          <div className="panel-heading">
            <div>
              <h3>Quick actions</h3>
              <p>Continue setting up your workspace</p>
            </div>
          </div>

          <button className="action-row" onClick={() => onNavigate("rules")}>
            <span className="action-icon violet">⌘</span>
            <span className="action-copy">
              <strong>Create a safety rule</strong>
              <small>Describe the rule in plain English</small>
            </span>
            <span className="action-arrow">→</span>
          </button>

          <button className="action-row" onClick={() => onNavigate("videos")}>
            <span className="action-icon blue">▣</span>
            <span className="action-copy">
              <strong>Upload recorded footage</strong>
              <small>Prepare a video for analysis</small>
            </span>
            <span className="action-arrow">→</span>
          </button>

          <button
            className="action-row"
            onClick={() => onNavigate("backtesting")}
          >
            <span className="action-icon amber">↻</span>
            <span className="action-copy">
              <strong>Explore backtesting</strong>
              <small>Plan a historical rule evaluation</small>
            </span>
            <span className="action-arrow">→</span>
          </button>

          <button className="action-row" onClick={() => onNavigate("review")}>
            <span className="action-icon green">☑</span>
            <span className="action-copy">
              <strong>Review potential events</strong>
              <small>Human verification and approval</small>
            </span>
            <span className="action-arrow">→</span>
          </button>
        </div>
      </section>

      <section className="panel activity-panel">
        <div className="panel-heading">
          <div>
            <h3>Recent activity</h3>
            <p>Actions recorded in this workspace</p>
          </div>
          <button className="text-button" onClick={() => onNavigate("audit")}>
            View audit history →
          </button>
        </div>
        <div className="empty-activity">
          <span className="activity-icon">◷</span>
          <div>
            <strong>No activity yet</strong>
            <p>
              Your rule creation, video analysis, and review history will appear
              here.
            </p>
          </div>
        </div>
      </section>
    </>
  );
}

function StatCard({
  label,
  value,
  note,
  icon,
  tone,
}: {
  label: string;
  value: string;
  note: string;
  icon: string;
  tone: string;
}) {
  return (
    <div className="stat-card">
      <div className="stat-top">
        <span>{label}</span>
        <span className={`stat-icon ${tone}`}>{icon}</span>
      </div>
      <div className="stat-value">{value}</div>
      <div className="stat-note">{note}</div>
    </div>
  );
}

type SafetyRuleDraft = {
  rule_name: string;
  description: string;
  hazard_type: string;
  detection_target: string;
  detector_id?: string | null;
  condition: string;
  condition_parameters?: Record<string, number | string | boolean>;
  required_capability_types?: string[];
  required_capabilities?: string[];
  severity: string;
  required_evidence: string[];
  recommended_action: string;
  confidence_threshold: number;
  zone_overlap_threshold?: number | null;
  limitations: string[];
};

type DetectorCapabilities = {
  targets: {
    class_id: number;
    class_name: string;
    detector_id: string;
    detector_name: string;
    detector_behaviors: string[];
  }[];
  conditions: {
    condition: string;
    spatial: boolean;
    required_target?: string;
    parameter?: string;
    required_capabilities?: string[];
    required_capability_types?: string[];
  }[];
  behaviors: string[];
  target_aliases: Record<string, string>;
  capabilities: {
    capability_id: string;
    detector_id: string;
    capability_type: string;
    description: string;
    supported_object_classes: string[];
    supported_attributes: string[];
    supported_states: string[];
    supported_behaviours: string[];
    supported_relationships: string[];
    required_inputs: string[];
    known_limitations: string[];
    validation_status: "unvalidated" | "validated" | "production_ready";
  }[];
  limitations: string[];
};

type ZonePoint = {
  x: number;
  y: number;
};

function ruleRequiresZone(rule: SafetyRuleDraft): boolean {
  const description = [
    rule.rule_name,
    rule.description,
    rule.detection_target,
    rule.condition,
    ...rule.required_evidence,
  ]
    .join(" ")
    .toLowerCase();

  return (
    rule.hazard_type === "restricted_area" ||
    [
      "restricted area",
      "restricted zone",
      "zone",
      "boundary",
      "enters the area",
      "enters an area",
      "entering",
      "crossing",
      "remaining within",
      "spatial boundaries",
      "inside the area",
      "within the area",
    ].some((term) => description.includes(term))
  );
}

type RuleValidationResult = {
  status: "invalid" | "needs_configuration" | "ready_for_evaluation";
  rule_name: string;
  message: string;
  findings: {
    check: string;
    status: "passed" | "failed" | "needs_configuration";
    message: string;
  }[];
};

type LiveFrameResult = {
  status: string;
  rule_name: string;
  target_class: string;
  detector_id: string;
  condition: string;
  timestamp_seconds: number;
  frame_width: number;
  frame_height: number;
  condition_met: boolean;
  evaluation_status: string;
  human_approval_required: boolean;
  detection_count: number;
  detections: (DetectionMatch & {
    class_id: number;
    detector_id: string;
    matches: boolean;
    zone_overlap_ratio?: number;
    zone_overlap_threshold?: number;
  })[];
  limitations: string[];
};

const exampleInstructions = [
  "Flag people entering a restricted area.",
  "Flag a detected object that exhibits a registered unsafe state.",
];

const supportedRestrictedAreaRule: SafetyRuleDraft = {
  rule_name: "Person in Restricted Zone",
  description:
    "Flag sampled video frames where a detected person is present inside the configured restricted zone.",
  hazard_type: "restricted_area",
  detection_target: "person",
  condition: "person_in_zone",
  severity: "high",
  required_evidence: ["person_detection", "restricted_zone", "timestamp"],
  recommended_action: "Review the matching frame and confirm the event.",
  confidence_threshold: 0.25,
  limitations: [
    "Evaluation checks sampled-frame zone presence; it does not detect continuous boundary crossing.",
    "A detected person inside the zone is potential evidence and requires human review.",
  ],
};

const supportedBoundingBoxOverlapRule: SafetyRuleDraft = {
  ...supportedRestrictedAreaRule,
  rule_name: "Person overlaps restricted zone",
  description:
    "Flag sampled frames where the configured fraction of a detected person's bounding box overlaps the restricted zone.",
  condition: "person_overlaps_zone",
  zone_overlap_threshold: 0.2,
  limitations: [
    "Overlap is polygon intersection area divided by person bounding-box area.",
    "The configured ratio is a geometric candidate only and requires human review.",
    "The detector's bounding box may be imprecise and does not identify body parts.",
  ],
};

function RuleBuilder({
  onNavigate,
  selectedSafetyRule,
  onSelectSafetyRule,
  configuredZone,
}: {
  onNavigate: (page: PageId) => void;
  selectedSafetyRule: SafetyRuleDraft | null;
  onSelectSafetyRule: (rule: SafetyRuleDraft | null) => void;
  configuredZone: ZonePoint[];
}) {
  const [instruction, setInstruction] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [rule, setRule] = useState<SafetyRuleDraft | null>(selectedSafetyRule);
  const [capabilities, setCapabilities] =
    useState<DetectorCapabilities | null>(null);
  const [capabilityError, setCapabilityError] = useState("");

  const [validation, setValidation] = useState<RuleValidationResult | null>(
    null,
  );
  const [validating, setValidating] = useState(false);

  useEffect(() => {
    let active = true;
    fetch(`${API_BASE_URL}/capabilities`)
      .then(async (response) => {
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || "Unable to load detector capabilities.");
        }
        if (active) setCapabilities(data as DetectorCapabilities);
      })
      .catch((err: unknown) => {
        if (active) {
          setCapabilityError(
            err instanceof Error
              ? err.message
              : "Unable to load detector capabilities.",
          );
        }
      });
    return () => {
      active = false;
    };
  }, []);

  async function generateRule() {
    setLoading(true);
    setError("");
    setRule(null);
    setValidation(null);
    onSelectSafetyRule(null);

    try {
      const response = await fetch(`${API_BASE_URL}/rules/generate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ instruction }),
      });

      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : JSON.stringify(data.detail || "Unable to generate a safety rule."),
        );
      }

      const generatedRule = data.rule as SafetyRuleDraft;
      setRule(generatedRule);
      onSelectSafetyRule(generatedRule);
    } catch (err) {
      setError(
        err instanceof TypeError && err.message === "Failed to fetch"
          ? `Could not reach the Rulevia backend at ${API_BASE_URL}. Check the backend URL, that the API is running, and that its CORS_ALLOWED_ORIGINS includes this frontend's origin. The restricted-area and overlap presets remain available without Gemini.`
          : err instanceof Error
            ? err.message
            : "Unable to generate a rule. The restricted-area and overlap presets remain available.",
      );
    } finally {
      setLoading(false);
    }
  }

  async function validateRule() {
    if (!rule) return;

    setValidating(true);
    setError("");
    setValidation(null);

    try {
      const response = await fetch(`${API_BASE_URL}/rules/validate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          rule,
          zone: configuredZone.length >= 3 ? configuredZone : null,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Rule validation failed.");
      }

      setValidation(data);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Unable to validate the draft.",
      );
    } finally {
      setValidating(false);
    }
  }

  function selectSupportedRestrictedAreaRule() {
    const supportedRule = { ...supportedRestrictedAreaRule };
    setInstruction(
      "Flag sampled frames where a person is present inside the configured restricted zone.",
    );
    setRule(supportedRule);
    setError("");
    setValidation(null);
    onSelectSafetyRule(supportedRule);
  }

  function selectSupportedBoundingBoxOverlapRule() {
    const supportedRule = { ...supportedBoundingBoxOverlapRule };
    setInstruction(
      "Flag a person when the configured portion of their bounding box overlaps the restricted zone.",
    );
    setRule(supportedRule);
    setError("");
    setValidation(null);
    onSelectSafetyRule(supportedRule);
  }

  return (
    <div className="rule-builder">
      <section className="panel rule-input-panel">
        <div className="rule-panel-heading">
          <div>
            <span className="rule-step-label">STEP 01 / INSTRUCTION</span>
            <h2>What should Rulevia look for?</h2>
            <p>
              Describe an observable safety condition in plain English. Gemini
              will turn it into a draft for your review.
            </p>
          </div>
          <span className="rule-ai-badge">✦ Gemini AI</span>
        </div>

        <label className="rule-field-label" htmlFor="safety-instruction">
          Safety instruction
        </label>

        <textarea
          id="safety-instruction"
          className="rule-textarea"
          value={instruction}
          onChange={(event) => {
            setInstruction(event.target.value);
            setRule(null);
            setValidation(null);
            onSelectSafetyRule(null);
          }}
          placeholder="Describe an object, visual state, spatial condition, or behavior to look for..."
          maxLength={1000}
          rows={5}
        />

        <div className="rule-input-footer">
          <span>{instruction.length}/1000 characters · Minimum 10</span>
          <span>Plain English</span>
        </div>

        <div className="rule-examples">
          <span className="rule-field-label">TRY AN EXAMPLE</span>
          <div className="example-buttons">
            {exampleInstructions.map((example) => (
              <button
                key={example}
                className="example-chip"
                onClick={() => {
                  setInstruction(example);
                  setError("");
                  setRule(null);
                  setValidation(null);
                  onSelectSafetyRule(null);
                }}
              >
                {example}
              </button>
            ))}
          </div>
        </div>

        <div className="rule-submit-row">
          <span className="rule-security-note">
            Uses supported person-in-zone evaluation; no AI generation required
          </span>
          <button
            type="button"
            className="secondary-button"
            onClick={selectSupportedRestrictedAreaRule}
          >
            Use supported restricted-area rule
          </button>
          <button
            type="button"
            className="secondary-button"
            onClick={selectSupportedBoundingBoxOverlapRule}
          >
            Use bounding-box overlap rule
          </button>
        </div>

        {error && (
          <div className="rule-error" role="alert">
            <strong>Rule Builder notice</strong>
            <p>{error}</p>
            <p>
              Natural-language generation is separate from video evaluation;
              the supported restricted-area and overlap presets remain usable.
            </p>
          </div>
        )}

        <div className="rule-submit-row">
          <span className="rule-security-note">
            API key stays on the backend
          </span>
          <button
            className="primary-button"
            disabled={loading || instruction.trim().length < 10}
            onClick={generateRule}
          >
            {loading ? "Generating draft…" : "✦ Generate safety rule"}
          </button>
        </div>
      </section>

      {loading && (
        <section className="panel rule-result-placeholder">
          <div className="loading-indicator" />
          <h3>Drafting your safety rule</h3>
          <p>Gemini is converting your instruction into structured fields.</p>
        </section>
      )}

      {rule && (
        <section className="panel generated-rule-panel">
          <div className="generated-rule-header">
            <div>
              <span className="rule-step-label">STEP 02 / GENERATED DRAFT</span>
              <h2>{rule.rule_name}</h2>
              <p>{rule.description}</p>
            </div>
            <span className="draft-status">Draft · Not validated</span>
          </div>

          <div className="generated-rule-grid">
            <div className="generated-field">
              <span>Hazard type</span>
              <strong>{rule.hazard_type.replaceAll("_", " ")}</strong>
            </div>
            <div className="generated-field">
              <span>Severity</span>
              <strong className={`severity-${rule.severity}`}>
                {rule.severity.toUpperCase()}
              </strong>
            </div>
            <div className="generated-field">
              <span>User-requested target</span>
              <strong>{rule.detection_target}</strong>
            </div>
            <div className="generated-field">
              <span>Proposed confidence threshold</span>
              <strong>{(rule.confidence_threshold * 100).toFixed(0)}%</strong>
            </div>
          </div>

          <div className="generated-section">
            <h3>Detector capability</h3>
            {capabilityError ? (
              <p role="alert">{capabilityError}</p>
            ) : !capabilities ? (
              <p>Checking registered detector targets…</p>
            ) : (
              (() => {
                const normalizedRequested = rule.detection_target
                  .trim()
                  .toLocaleLowerCase()
                  .replaceAll("_", " ");
                const canonicalRequested =
                  capabilities.target_aliases[normalizedRequested] ??
                  normalizedRequested;
                const selectedTarget = capabilities.targets.find(
                  (target) =>
                    target.class_name.toLocaleLowerCase() === canonicalRequested &&
                    (!rule.detector_id ||
                      target.detector_id === rule.detector_id),
                );
                const eligibleConditions = capabilities.conditions.filter(
                  (item) =>
                    !item.required_target ||
                    item.required_target === selectedTarget?.class_name,
                );
                const conditionCapability = capabilities.conditions.find(
                  (item) => item.condition === rule.condition,
                );
                const supportedCondition =
                  Boolean(conditionCapability) &&
                  (!conditionCapability?.required_target ||
                    conditionCapability.required_target ===
                      selectedTarget?.class_name);
                const registeredCapabilityTypes = new Set(
                  capabilities.capabilities.map(
                    (capability) => capability.capability_type,
                  ),
                );
                const requestedCapabilityTypes =
                  rule.required_capability_types ?? [];
                const missingCapabilityTypes = requestedCapabilityTypes.filter(
                  (type) => !registeredCapabilityTypes.has(type),
                );
                const requestedCapabilityIds = [
                  ...new Set([
                    ...(rule.required_capabilities ?? []),
                    ...(conditionCapability?.required_capabilities ?? []),
                  ]),
                ];
                const registeredCapabilityIds = new Set(
                  capabilities.capabilities.map(
                    (capability) => capability.capability_id,
                  ),
                );
                const missingCapabilityIds = requestedCapabilityIds.filter(
                  (id) => !registeredCapabilityIds.has(id),
                );
                return (
                  <>
                    <div className="generated-field">
                      <span>Executable target</span>
                      <strong>
                        {selectedTarget
                          ? `${selectedTarget.class_name} · ${selectedTarget.detector_name}`
                          : "Not currently available"}
                      </strong>
                    </div>
                    <label className="rule-field-label" htmlFor="detector-target">
                      Choose an available detector target
                    </label>
                    <select
                      id="detector-target"
                      value={
                        selectedTarget
                          ? `${selectedTarget.detector_id}:${selectedTarget.class_id}`
                          : ""
                      }
                      onChange={(event) => {
                        const target = capabilities.targets.find(
                          (item) =>
                            `${item.detector_id}:${item.class_id}` ===
                            event.target.value,
                        );
                        if (!target) return;
                        const updatedRule = {
                          ...rule,
                          detection_target: target.class_name,
                          detector_id: target.detector_id,
                        };
                        setRule(updatedRule);
                        onSelectSafetyRule(updatedRule);
                        setValidation(null);
                      }}
                    >
                      <option value="">Keep requested target (not executable)</option>
                      {capabilities.targets.map((target) => (
                        <option
                          key={`${target.detector_id}:${target.class_id}`}
                          value={`${target.detector_id}:${target.class_id}`}
                        >
                          {target.class_name} — {target.detector_name}
                        </option>
                      ))}
                    </select>
                    <p>
                      Condition: {supportedCondition ? "supported" : "unsupported"}.
                      {" "}A language-model draft cannot add missing visual
                      detection capabilities.
                    </p>
                    {requestedCapabilityTypes.length > 0 && (
                      <div className="capability-requirements">
                        <strong>Requested capability types</strong>
                        <p>
                          {requestedCapabilityTypes.join(", ")}
                          {missingCapabilityTypes.length > 0
                            ? ` · Missing: ${missingCapabilityTypes.join(", ")}`
                            : " · Registered"}
                        </p>
                      </div>
                    )}
                    {requestedCapabilityIds.length > 0 && (
                      <div className="capability-requirements">
                        <strong>Requested capability contracts</strong>
                        <p>
                          {requestedCapabilityIds.join(", ")}
                          {missingCapabilityIds.length > 0
                            ? ` · Missing: ${missingCapabilityIds.join(", ")}`
                            : " · Registered"}
                        </p>
                      </div>
                    )}
                    <details className="capability-catalog">
                      <summary>Registered visual capability contracts</summary>
                      {capabilities.capabilities.map((capability) => (
                        <article
                          className="capability-catalog-item"
                          key={capability.capability_id}
                        >
                          <strong>
                            {capability.capability_id} · {capability.capability_type}
                          </strong>
                          <span>
                            {capability.validation_status.replaceAll("_", " ")}
                          </span>
                          <p>{capability.description}</p>
                          {capability.supported_object_classes.length > 0 && (
                            <p>
                              Classes:{" "}
                              {capability.supported_object_classes.join(", ")}
                            </p>
                          )}
                          {(capability.supported_attributes.length > 0 ||
                            capability.supported_states.length > 0 ||
                            capability.supported_behaviours.length > 0 ||
                            capability.supported_relationships.length > 0) && (
                            <p>
                              Attributes: {capability.supported_attributes.join(", ") || "none"} ·
                              States: {capability.supported_states.join(", ") || "none"} ·
                              Behaviours: {capability.supported_behaviours.join(", ") || "none"} ·
                              Relationships: {capability.supported_relationships.join(", ") || "none"}
                            </p>
                          )}
                          {capability.known_limitations.length > 0 && (
                            <p>
                              Limitations: {capability.known_limitations.join(" ")}
                            </p>
                          )}
                        </article>
                      ))}
                    </details>
                    <label className="rule-field-label" htmlFor="rule-condition">
                      Evaluation condition
                    </label>
                    <select
                      id="rule-condition"
                      value={
                        eligibleConditions.some(
                          (item) => item.condition === rule.condition,
                        )
                          ? rule.condition
                          : ""
                      }
                      onChange={(event) => {
                        if (!event.target.value) return;
                        const updatedRule = {
                          ...rule,
                          condition: event.target.value,
                        };
                        setRule(updatedRule);
                        onSelectSafetyRule(updatedRule);
                        setValidation(null);
                      }}
                    >
                      <option value="">
                        Keep requested condition (not currently executable)
                      </option>
                      {eligibleConditions.map((item) => (
                        <option key={item.condition} value={item.condition}>
                          {item.condition}
                        </option>
                      ))}
                    </select>
                  </>
                );
              })()
            )}
          </div>

          <div className="generated-section">
            <h3>Trigger condition</h3>
            <p>
              {rule.condition === "person_in_zone"
                ? "A detected person is present inside the configured restricted zone in a sampled frame."
                : rule.condition === "person_overlaps_zone"
                  ? "The polygon intersection area divided by the person's bounding-box area meets the configured threshold."
                : rule.condition}
            </p>
            {[
              "person_overlaps_zone",
              "object_overlaps_zone",
            ].includes(rule.condition) && (
              <div className="generated-field">
                <label htmlFor="zone-overlap-threshold">
                  Minimum bounding-box overlap ratio (greater than 0, up to 1)
                </label>
                <input
                  id="zone-overlap-threshold"
                  type="number"
                  min="0.01"
                  max="1"
                  step="0.01"
                  value={
                    rule.zone_overlap_threshold ??
                    (typeof rule.condition_parameters?.overlap_threshold ===
                      "number" ||
                    typeof rule.condition_parameters?.overlap_threshold ===
                      "string"
                      ? rule.condition_parameters.overlap_threshold
                      : "")
                  }
                  onChange={(event) => {
                    const rawValue = event.target.value;
                    const threshold =
                      rawValue === "" ? undefined : Number(rawValue);
                    const conditionParameters = {
                      ...rule.condition_parameters,
                    };
                    if (threshold === undefined) {
                      delete conditionParameters.overlap_threshold;
                    } else {
                      conditionParameters.overlap_threshold = threshold;
                    }
                    const updatedRule = {
                      ...rule,
                      zone_overlap_threshold: threshold,
                      condition_parameters: conditionParameters,
                    };
                    setRule(updatedRule);
                    onSelectSafetyRule(updatedRule);
                    setValidation(null);
                  }}
                />
              </div>
            )}
          </div>

          <div className="generated-section">
            <h3>Evidence to check</h3>
            <ul>
              {rule.required_evidence.map((item, index) => (
                <li key={`${item}-${index}`}>{item}</li>
              ))}
            </ul>
          </div>

          <div className="generated-section">
            <h3>Recommended reviewer action</h3>
            <p>{rule.recommended_action}</p>
          </div>

          <div className="generated-section limitations-section">
            <h3>Limitations and human review</h3>
            <ul>
              {rule.limitations.map((item, index) => (
                <li key={`${item}-${index}`}>{item}</li>
              ))}
            </ul>
          </div>

          <div className="draft-warning">
            <strong>Important:</strong> This is an AI-generated draft, not an
            approved safety control. Validate its conditions and detector
            capabilities before using it for monitoring.
          </div>

          {validation && (
            <div className={`validation-panel validation-${validation.status}`}>
              <div className="validation-heading">
                <div>
                  <span className="rule-step-label">RULE VALIDATION</span>
                  <h3>
                    {validation.status === "ready_for_evaluation"
                      ? "Preliminary checks passed"
                      : validation.status === "needs_configuration"
                        ? "Configuration required"
                        : "Rule needs changes"}
                  </h3>
                </div>
                <span className="validation-status-label">
                  {validation.status.replaceAll("_", " ")}
                </span>
              </div>

              <p className="validation-message">{validation.message}</p>

              {validation.findings.map((finding, index) => (
                <div
                  className="validation-finding"
                  key={`${finding.check}-${index}`}
                >
                  <span
                    className={`finding-dot finding-${finding.status}`}
                    aria-hidden="true"
                  />
                  <div>
                    <strong>{finding.check.replaceAll("_", " ")}</strong>
                    <p>{finding.message}</p>
                  </div>
                </div>
              ))}
            </div>
          )}

          <div className="generated-rule-footer">
            <button
              className="primary-button"
              disabled={validating}
              onClick={validateRule}
            >
              {validating ? "Validating…" : "✓ Validate draft"}
            </button>

            <button
              className="secondary-button"
              onClick={() => {
                setRule(null);
                setError("");
                setValidation(null);
                onSelectSafetyRule(null);
              }}
            >
              Generate another draft
            </button>
            <button
              className="primary-button"
              onClick={() => onNavigate("videos")}
            >
              Continue to video library →
            </button>
          </div>
        </section>
      )}
    </div>
  );
}

type ExtractedFrame = {
  frame_number: number;
  timestamp_seconds: number;
  image_url: string;
};

type VideoUploadResult = {
  status: string;
  video_id: string;
  filename: string;
  duration_seconds: number;
  frame_interval_seconds: number;
  total_frames_extracted: number;
  frames: ExtractedFrame[];
};

type DetectionMatch = {
  class_name: string;
  confidence: number;
  box: [number, number, number, number];
  anchor_point?: ZonePoint;
  inside_zone?: boolean;
};

type EntryTransitionDiagnostic = {
  detection_count: number;
  observations: {
    reference_point: ZonePoint;
    inside_zone: boolean;
  }[];
  transition_evaluated: boolean;
  transition_result: "entry_candidate" | "no_entry" | "not_evaluated";
  reason: string;
};

type ZonePresenceDiagnostic = {
  detection_count: number;
  observations: {
    reference_point: ZonePoint;
    inside_zone: boolean;
  }[];
  matching_detection_count: number;
  matching_status: boolean;
};

type ZoneOverlapDiagnostic = {
  detection_count: number;
  threshold: number;
  observations: {
    box: [number, number, number, number];
    overlap_ratio: number;
    matches: boolean;
  }[];
  matching_detection_count: number;
  matching_status: boolean;
};

type EvaluatedFrame = {
  frame: string;
  timestamp_seconds: number;
  frame_width: number;
  frame_height: number;
  target_observed: boolean;
  zone_match: boolean | null;
  zone_diagnostic: ZonePresenceDiagnostic | null;
  overlap_diagnostic: ZoneOverlapDiagnostic | null;
  entry_transition: EntryTransitionDiagnostic | null;
  detections: DetectionMatch[];
  matches: DetectionMatch[];
};

type VideoRuleEvaluationResult = {
  status: string;
  video_id: string;
  rule_name: string;
  target_class: string;
  condition: string;
  condition_met: boolean;
  evaluation_status: string;
  human_approval_required: boolean;
  total_frames_evaluated: number;
  frames_with_target: number;
  frames_matching_condition: number;
  zone: ZonePoint[] | null;
  zone_overlap_threshold: number | null;
  matching_frames: EvaluatedFrame[];
  frame_results: EvaluatedFrame[];
  limitations: string[];
};

function VideoLibrary({
  selectedSafetyRule,
  result,
  onVideoResultChange,
  configuredZone,
  onConfiguredZoneChange,
}: {
  selectedSafetyRule: SafetyRuleDraft | null;
  result: VideoUploadResult | null;
  onVideoResultChange: (result: VideoUploadResult | null) => void;
  configuredZone: ZonePoint[];
  onConfiguredZoneChange: (zone: ZonePoint[]) => void;
}) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const [zoneFrameNumber, setZoneFrameNumber] = useState<number | null>(
    result?.frames[0]?.frame_number ?? null,
  );
  const [drawingZone, setDrawingZone] = useState(false);
  const [draftZone, setDraftZone] = useState<ZonePoint[]>([]);

  const [reviewingFrame, setReviewingFrame] = useState("");
  const [reviewMessage, setReviewMessage] = useState("");
  const [reviewError, setReviewError] = useState("");
  const [reviewMessageFrame, setReviewMessageFrame] = useState("");

  const [evaluating, setEvaluating] = useState(false);
  const [evaluationError, setEvaluationError] = useState("");
  const [evaluationResult, setEvaluationResult] =
    useState<VideoRuleEvaluationResult | null>(null);

  async function uploadVideo() {
    if (!selectedFile) {
      setError("Please select a video first.");
      return;
    }

    setUploading(true);
    setError("");
    onVideoResultChange(null);
    onConfiguredZoneChange([]);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch(
        `${API_BASE_URL}/videos/upload-extract`,
        {
          method: "POST",
          body: formData,
        },
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Video upload failed.");
      }

      const uploadResult = data as VideoUploadResult;
      onVideoResultChange(uploadResult);
      setZoneFrameNumber(uploadResult.frames[0]?.frame_number ?? null);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to upload the video. Check the backend.",
      );
    } finally {
      setUploading(false);
    }
  }

  async function submitReview(
    frame: EvaluatedFrame,
    match: DetectionMatch,
    decision: "confirmed" | "dismissed" | "needs_further_review",
  ) {
    const reviewKey = `${frame.frame}-${decision}`;
    setReviewMessageFrame(frame.frame);

    setReviewingFrame(reviewKey);
    setReviewMessage("");
    setReviewError("");

    try {
      const response = await fetch(`${API_BASE_URL}/reviews`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          video_id: result?.video_id,
          frame: frame.frame,
          timestamp_seconds: frame.timestamp_seconds,
          target_class: match.class_name,
          confidence: match.confidence,
          decision,
          reviewer: "dashboard-reviewer",
          notes: "",
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : JSON.stringify(data.detail || "Unable to save review."),
        );
      }

      setReviewMessage(
        `Review saved: ${decision.replaceAll("_", " ")} for ${frame.frame}`,
      );
    } catch (err) {
      setReviewError(
        err instanceof Error ? err.message : "Unable to save review.",
      );
    } finally {
      setReviewingFrame("");
    }
  }

  async function evaluateSafetyRule() {
    if (!result?.video_id) {
      setEvaluationError("Upload a video first.");
      return;
    }

    setEvaluating(true);
    setEvaluationError("");
    setEvaluationResult(null);

    if (!selectedSafetyRule) {
      setEvaluationError(
        "Please generate a safety rule in Rule Builder first.",
      );
      setEvaluating(false);
      return;
    }

    const rule = selectedSafetyRule;

    try {
      const response = await fetch(
        `${API_BASE_URL}/rules/evaluate-video`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            video_id: result.video_id,
            rule,
            zone: configuredZone.length >= 3 ? configuredZone : null,
          }),
        },
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : JSON.stringify(data.detail || "Evaluation failed."),
        );
      }

      setEvaluationResult(data as VideoRuleEvaluationResult);
    } catch (err) {
      setEvaluationError(
        err instanceof Error
          ? err.message
          : "Unable to evaluate the safety rule.",
      );
    } finally {
      setEvaluating(false);
    }
  }



  function formatTimestamp(seconds: number) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);

    return `${String(minutes).padStart(2, "0")}:${String(
      remainingSeconds,
    ).padStart(2, "0")}`;
  }

  function handleZoneDrawingClick(event: MouseEvent<SVGSVGElement>) {
    if (!drawingZone) return;

    const bounds = event.currentTarget.getBoundingClientRect();
    const point = {
      x: Math.min(Math.max((event.clientX - bounds.left) / bounds.width, 0), 1),
      y: Math.min(Math.max((event.clientY - bounds.top) / bounds.height, 0), 1),
    };
    const updatedZone = [...draftZone, point];
    setDraftZone(updatedZone);
  }

  const zoneFrame =
    result?.frames.find((frame) => frame.frame_number === zoneFrameNumber) ??
    result?.frames[0] ??
    null;
  const spatialRule = selectedSafetyRule
    ? ruleRequiresZone(selectedSafetyRule)
    : false;
  const evaluatedCondition = evaluationResult?.condition.trim().toLowerCase();
  const pointsForDisplay = drawingZone ? draftZone : configuredZone;
  const svgPoints = pointsForDisplay
    .map((point) => `${point.x * 1000},${point.y * 1000}`)
    .join(" ");

  return (
    <div className="video-library">
      <section className="panel video-upload-panel">
        <div className="panel-heading">
          <div>
            <h3>Upload CCTV footage</h3>
            <p>
              Upload a recorded video to extract timestamped frames for safety
              analysis.
            </p>
          </div>
          <span className="rule-ai-badge">LOCAL PROCESSING</span>
        </div>

        <label className="rule-field-label" htmlFor="cctv-video">
          Select video
        </label>

        <input
          id="cctv-video"
          type="file"
          accept=".mp4,.avi,.mov,.mkv,video/*"
          onChange={(event) => {
            setSelectedFile(event.target.files?.[0] ?? null);
            onVideoResultChange(null);
            onConfiguredZoneChange([]);
            setDraftZone([]);
            setDrawingZone(false);
            setError("");
          }}
        />

        <p className="video-file-help">
          Supported formats: MP4, AVI, MOV, MKV. Maximum size: 100 MB.
        </p>

        {selectedFile && (
          <p className="video-selected-file">
            Selected: {selectedFile.name} (
            {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB)
          </p>
        )}

        {error && (
          <div className="video-error" role="alert">
            {error}
          </div>
        )}

        <button
          className="primary-button"
          onClick={uploadVideo}
          disabled={!selectedFile || uploading}
        >
          {uploading
            ? "Uploading and extracting…"
            : "Upload and extract frames →"}
        </button>
      </section>

      {result && (
        <>
          <section className="stats-grid video-result-stats">
            <div className="stat-card">
              <div className="stat-top">
                <span>Video duration</span>
              </div>
              <div className="stat-value">
                {formatTimestamp(result.duration_seconds)}
              </div>
              <div className="stat-note">Minutes:seconds</div>
            </div>

            <div className="stat-card">
              <div className="stat-top">
                <span>Frames extracted</span>
              </div>
              <div className="stat-value">{result.total_frames_extracted}</div>
              <div className="stat-note">
                One frame every {result.frame_interval_seconds} seconds
              </div>
            </div>

            <div className="stat-card">
              <div className="stat-top">
                <span>Upload status</span>
              </div>
              <div className="stat-value">Success</div>
              <div className="stat-note">
                Video ID: {result.video_id.slice(0, 8)}
              </div>
            </div>
          </section>

          <section className="panel extracted-frames-panel">
            <div className="panel-heading">
              <div>
                <h3>Extracted frames</h3>
                <p>{result.filename}</p>
              </div>
            </div>

            <div className="extracted-frames-grid">
              {result.frames.map((frame) => (
                <article
                  className="extracted-frame-card"
                  key={frame.frame_number}
                >
                  <img
                    src={`${API_BASE_URL}${frame.image_url}`}
                    alt={`Extracted frame at ${formatTimestamp(
                      frame.timestamp_seconds,
                    )}`}
                    loading="lazy"
                  />

                  <div className="extracted-frame-info">
                    <strong>Frame {frame.frame_number}</strong>
                    <span>
                      Timestamp: {formatTimestamp(frame.timestamp_seconds)}
                    </span>
                  </div>
                </article>
              ))}
            </div>
          </section>

          <section className="panel zone-configuration-panel">
            <div className="panel-heading">
              <div>
                <h3>Restricted-zone configuration</h3>
                <p>
                  Select a representative frame and mark the area to monitor.
                  Coordinates are normalized to the video frame.
                </p>
              </div>
              <span
                className={`zone-status-pill ${
                  configuredZone.length >= 3 ? "zone-status-ready" : ""
                }`}
              >
                {drawingZone
                  ? "Drawing zone"
                  : configuredZone.length >= 3
                    ? "Zone configured"
                    : spatialRule
                      ? "Required for this rule"
                      : "Optional"}
              </span>
            </div>

            <label className="rule-field-label" htmlFor="zone-reference-frame">
              Reference frame
            </label>
            <select
              id="zone-reference-frame"
              className="zone-frame-select"
              value={zoneFrameNumber ?? ""}
              onChange={(event) =>
                setZoneFrameNumber(
                  event.target.value ? Number(event.target.value) : null,
                )
              }
            >
              {result.frames.map((frame) => (
                <option key={frame.frame_number} value={frame.frame_number}>
                  Frame {frame.frame_number} ·{" "}
                  {formatTimestamp(frame.timestamp_seconds)}
                </option>
              ))}
            </select>

            {zoneFrame && (
              <div className="zone-editor-stage">
                <img
                  src={`${API_BASE_URL}${zoneFrame.image_url}`}
                  alt={`Zone reference frame ${zoneFrame.frame_number}`}
                />
                <svg
                  className={`zone-drawing-overlay ${
                    drawingZone ? "zone-drawing-active" : ""
                  }`}
                  viewBox="0 0 1000 1000"
                  preserveAspectRatio="none"
                  aria-label="Restricted-zone polygon editor"
                  onClick={handleZoneDrawingClick}
                >
                  {pointsForDisplay.length >= 3 && !drawingZone && (
                    <polygon points={svgPoints} className="zone-polygon" />
                  )}
                  {pointsForDisplay.length > 0 && (
                    <polyline
                      points={svgPoints}
                      className="zone-polyline"
                    />
                  )}
                  {pointsForDisplay.map((point, index) => (
                    <circle
                      key={`zone-point-${index}`}
                      cx={point.x * 1000}
                      cy={point.y * 1000}
                      r="9"
                      className="zone-vertex"
                    />
                  ))}
                </svg>
              </div>
            )}

            <p className="zone-editor-help">
              {drawingZone
                ? `Click the frame to add points (${draftZone.length} placed). Finish with at least 3 points.`
                : configuredZone.length >= 3
                  ? "A detection is considered in the zone when the bottom-center of its box lies inside the polygon."
                  : "A restricted-area rule needs a polygon. A detection matches when the bottom-center of its box lies inside it."}
            </p>
            <div className="zone-editor-actions">
              {drawingZone ? (
                <>
                  <button
                    type="button"
                    className="secondary-button"
                    onClick={() => setDraftZone((points) => points.slice(0, -1))}
                    disabled={draftZone.length === 0}
                  >
                    Undo last point
                  </button>
                  <button
                    type="button"
                    className="primary-button"
                    onClick={() => {
                      onConfiguredZoneChange(draftZone);
                      setDrawingZone(false);
                    }}
                    disabled={draftZone.length < 3}
                  >
                    Finish zone
                  </button>
                  <button
                    type="button"
                    className="secondary-button"
                    onClick={() => {
                      setDraftZone([]);
                      setDrawingZone(false);
                    }}
                  >
                    Cancel
                  </button>
                </>
              ) : (
                <button
                  type="button"
                  className="secondary-button"
                  onClick={() => {
                    setDraftZone([]);
                    setDrawingZone(true);
                  }}
                >
                  {configuredZone.length >= 3 ? "Redraw zone" : "Draw zone"}
                </button>
              )}
            </div>
          </section>

          <section className="panel evaluation-panel">
            <div className="panel-heading">
              <div>
                <h3>Safety Rule Evaluation</h3>
                <p>
                  {selectedSafetyRule
                    ? `Evaluate “${selectedSafetyRule.rule_name}” against this video.`
                    : "Generate a rule in Rule Builder before evaluating this video."}
                </p>
              </div>
            </div>

            <button
              type="button"
              className="primary-button"
              onClick={evaluateSafetyRule}
              disabled={
                evaluating ||
                !selectedSafetyRule ||
                (spatialRule && configuredZone.length < 3)
              }
            >
              {evaluating ? "Evaluating video..." : "Evaluate Safety Rule"}
            </button>

            {spatialRule && configuredZone.length < 3 && (
              <p className="stat-note">
                Configure a restricted zone above to enable this spatial rule.
              </p>
            )}

            {evaluationError && (
              <p role="alert" className="error-message">
                {evaluationError}
              </p>
            )}

            {evaluationResult && (
              <div className="evaluation-results">
                <h3>Evaluation Results</h3>

                <p>
                  <strong>Rule:</strong> {evaluationResult.rule_name}
                </p>

                <p>
                  <strong>Condition key:</strong> {evaluationResult.condition}
                </p>

                <p>
                  <strong>Condition met:</strong>{" "}
                  {String(evaluationResult.condition_met)}
                </p>

                <p>
                  <strong>Frames evaluated:</strong>{" "}
                  {evaluationResult.total_frames_evaluated}
                </p>

                <p>
                  <strong>Frames with a person:</strong>{" "}
                  {evaluationResult.frames_with_target}
                </p>

                <p>
                  <strong>Frames matching condition:</strong>{" "}
                  {evaluationResult.frames_matching_condition}
                </p>

                <p>
                  <strong>Review status:</strong>{" "}
                  {evaluationResult.evaluation_status}
                </p>

                {evaluatedCondition === "person_in_zone" && (
                  <>
                    <h4>Zone Presence by Frame</h4>
                    <div style={{ overflowX: "auto" }}>
                      <table>
                        <thead>
                          <tr>
                            <th>Frame</th>
                            <th>Timestamp</th>
                            <th>People detected</th>
                            <th>Reference point and zone result</th>
                            <th>Matching detections</th>
                            <th>Frame matches</th>
                          </tr>
                        </thead>
                        <tbody>
                          {evaluationResult.frame_results.map((frame) => {
                            const diagnostic = frame.zone_diagnostic;
                            return (
                              <tr key={frame.frame}>
                                <td>{frame.frame}</td>
                                <td>
                                  {formatTimestamp(frame.timestamp_seconds)}
                                </td>
                                <td>{diagnostic?.detection_count ?? 0}</td>
                                <td>
                                  {diagnostic?.observations.length
                                    ? diagnostic.observations.map(
                                        (observation, index) => (
                                          <div key={`${frame.frame}-${index}`}>
                                            (
                                            {observation.reference_point.x.toFixed(
                                              3,
                                            )}
                                            ,{" "}
                                            {observation.reference_point.y.toFixed(
                                              3,
                                            )}
                                            ){" "}
                                            {observation.inside_zone
                                              ? "inside"
                                              : "outside"}
                                          </div>
                                        ),
                                      )
                                    : "No person detected"}
                                </td>
                                <td>
                                  {diagnostic?.matching_detection_count ?? 0}
                                </td>
                                <td>
                                  {diagnostic?.matching_status
                                    ? "Yes"
                                    : "No"}
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </>
                )}

                {evaluatedCondition === "person_enters_zone" && (
                  <>
                    <h4>Entry Transition Diagnostics</h4>
                    <div style={{ overflowX: "auto" }}>
                      <table>
                        <thead>
                          <tr>
                            <th>Frame</th>
                            <th>Timestamp</th>
                            <th>People detected</th>
                            <th>Reference point and zone result</th>
                            <th>Transition evaluation</th>
                          </tr>
                        </thead>
                        <tbody>
                          {evaluationResult.frame_results.map((frame) => {
                            const diagnostic = frame.entry_transition;
                            return (
                              <tr key={frame.frame}>
                                <td>{frame.frame}</td>
                                <td>
                                  {formatTimestamp(frame.timestamp_seconds)}
                                </td>
                                <td>{diagnostic?.detection_count ?? 0}</td>
                                <td>
                                  {diagnostic?.observations.length
                                    ? diagnostic.observations.map(
                                        (observation, index) => (
                                          <div key={`${frame.frame}-${index}`}>
                                            (
                                            {observation.reference_point.x.toFixed(
                                              3,
                                            )}
                                            ,{" "}
                                            {observation.reference_point.y.toFixed(
                                              3,
                                            )}
                                            ){" "}
                                            {observation.inside_zone
                                              ? "inside"
                                              : "outside"}
                                          </div>
                                        ),
                                      )
                                    : "No person detected"}
                                </td>
                                <td>
                                  {diagnostic
                                    ? `${diagnostic.transition_result.replaceAll(
                                        "_",
                                        " ",
                                      )}: ${diagnostic.reason.replaceAll(
                                        "_",
                                        " ",
                                      )}`
                                    : "No entry diagnostics"}
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </>
                )}

                {evaluatedCondition === "person_overlaps_zone" && (
                  <>
                    <h4>Bounding-Box / Zone Overlap by Frame</h4>
                    <p>
                      Match means polygon intersection area divided by
                      bounding-box area is at least{" "}
                      {evaluationResult.zone_overlap_threshold}.
                    </p>
                    <div style={{ overflowX: "auto" }}>
                      <table>
                        <thead>
                          <tr>
                            <th>Frame</th>
                            <th>Timestamp</th>
                            <th>People detected</th>
                            <th>Bounding box (x1, y1, x2, y2)</th>
                            <th>Overlap ratio</th>
                            <th>Threshold</th>
                            <th>Detection matches</th>
                          </tr>
                        </thead>
                        <tbody>
                          {evaluationResult.frame_results.map((frame) => {
                            const diagnostic = frame.overlap_diagnostic;
                            return (
                              <tr key={frame.frame}>
                                <td>{frame.frame}</td>
                                <td>
                                  {formatTimestamp(frame.timestamp_seconds)}
                                </td>
                                <td>{diagnostic?.detection_count ?? 0}</td>
                                <td>
                                  {diagnostic?.observations.length
                                    ? diagnostic.observations.map(
                                        (observation, index) => (
                                          <div key={`${frame.frame}-${index}`}>
                                            {observation.box
                                              .map((coordinate) =>
                                                coordinate.toFixed(1),
                                              )
                                              .join(", ")}
                                          </div>
                                        ),
                                      )
                                    : "No person detected"}
                                </td>
                                <td>
                                  {diagnostic?.observations.length
                                    ? diagnostic.observations.map(
                                        (observation, index) => (
                                          <div key={`${frame.frame}-${index}`}>
                                            {observation.overlap_ratio.toFixed(
                                              3,
                                            )}
                                          </div>
                                        ),
                                      )
                                    : "—"}
                                </td>
                                <td>
                                  {diagnostic?.threshold.toFixed(3) ?? "—"}
                                </td>
                                <td>
                                  {diagnostic?.observations.length
                                    ? diagnostic.observations.map(
                                        (observation, index) => (
                                          <div key={`${frame.frame}-${index}`}>
                                            {observation.matches ? "Yes" : "No"}
                                          </div>
                                        ),
                                      )
                                    : "No"}
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </>
                )}

                <h4>Matching Frames</h4>

                {evaluationResult.matching_frames?.length > 0 ? (
                  <div className="extracted-frames-grid">
                    {evaluationResult.matching_frames.map((frame) => (
                      <article
                        className="extracted-frame-card"
                        key={frame.frame}
                      >
                        <div
                          style={{
                            position: "relative",
                            width: "100%",
                            overflow: "hidden",
                            borderRadius: "8px",
                          }}
                        >
                          <img
                            src={`${API_BASE_URL}/frames/${result.video_id}/${frame.frame}`}
                            alt={`Detected person at ${formatTimestamp(
                              frame.timestamp_seconds,
                            )}`}
                            loading="lazy"
                            style={{
                              display: "block",
                              width: "100%",
                              height: "auto",
                            }}
                          />

                          {evaluationResult.zone && (
                            <svg
                              className="evidence-zone-overlay"
                              viewBox="0 0 1000 1000"
                              preserveAspectRatio="none"
                              aria-hidden="true"
                            >
                              <polygon
                                points={evaluationResult.zone
                                  .map(
                                    (point) =>
                                      `${point.x * 1000},${point.y * 1000}`,
                                  )
                                  .join(" ")}
                                className="evidence-zone-polygon"
                              />
                              {frame.matches.map(
                                (match, index) =>
                                  match.anchor_point && (
                                    <circle
                                      key={`anchor-${index}`}
                                      cx={match.anchor_point.x * 1000}
                                      cy={match.anchor_point.y * 1000}
                                      r="10"
                                      className="evidence-zone-anchor"
                                    />
                                  ),
                              )}
                            </svg>
                          )}

                          {frame.matches
                            .map((match, index) => {
                              const [x1, y1, x2, y2] = match.box;
                              const width = frame.frame_width;
                              const height = frame.frame_height;

                              return (
                                <div
                                  key={`${frame.frame}-${index}`}
                                  style={{
                                    position: "absolute",
                                    left: `${(x1 / width) * 100}%`,
                                    top: `${(y1 / height) * 100}%`,
                                    width: `${((x2 - x1) / width) * 100}%`,
                                    height: `${((y2 - y1) / height) * 100}%`,
                                    border: "2px solid #22c55e",
                                    boxSizing: "border-box",
                                    pointerEvents: "none",
                                  }}
                                >
                                  <span
                                    style={{
                                      position: "absolute",
                                      left: "-2px",
                                      top: "-23px",
                                      background: "#22c55e",
                                      color: "#07120a",
                                      padding: "3px 6px",
                                      fontSize: "12px",
                                      fontWeight: 700,
                                      whiteSpace: "nowrap",
                                    }}
                                  >
                                    {match.class_name}{" "}
                                    {(match.confidence * 100).toFixed(1)}%
                                  </span>
                                </div>
                              );
                            })}

                          {reviewMessage &&
                            reviewMessageFrame === frame.frame && (
                              <p style={{ color: "green", marginTop: "12px" }}>
                                {reviewMessage}
                              </p>
                            )}

                          {reviewError &&
                            reviewMessageFrame === frame.frame && (
                              <p style={{ color: "red", marginTop: "12px" }}>
                                {reviewError}
                              </p>
                            )}
                        </div>

                        <div className="extracted-frame-info">
                          <strong>{frame.frame}</strong>

                          <span>
                            Timestamp:{" "}
                            {formatTimestamp(frame.timestamp_seconds)}
                          </span>

                          {frame.matches.map((match, index) => (
                            <div
                              key={`${frame.frame}-${index}`}
                              className="review-controls"
                            >
                              <p>
                                {match.class_name} —{" "}
                                {(match.confidence * 100).toFixed(1)}%
                                confidence
                              </p>

                              <button
                                type="button"
                                disabled={!!reviewingFrame}
                                onClick={() =>
                                  submitReview(frame, match, "confirmed")
                                }
                              >
                                {reviewingFrame === `${frame.frame}-confirmed`
                                  ? "Saving..."
                                  : "Confirm finding"}
                              </button>

                              <button
                                type="button"
                                disabled={!!reviewingFrame}
                                onClick={() =>
                                  submitReview(frame, match, "dismissed")
                                }
                              >
                                Dismiss finding
                              </button>

                              <button
                                type="button"
                                disabled={!!reviewingFrame}
                                onClick={() =>
                                  submitReview(
                                    frame,
                                    match,
                                    "needs_further_review",
                                  )
                                }
                              >
                                Needs further review
                              </button>
                            </div>
                          ))}
                        </div>
                      </article>
                    ))}
                  </div>
                ) : (
                  <p>No matching frames were found.</p>
                )}

                <p className="stat-note">
                  Detection results require human review and do not
                  independently confirm a safety violation.
                </p>
              </div>
            )}
          </section>
        </>
      )}
    </div>
  );
}

type ReviewRecord = {
  review_id: string;
  video_id: string;
  frame: string;
  timestamp_seconds: number;
  target_class: string;
  confidence: number;
  decision: "confirmed" | "dismissed" | "needs_further_review";
  reviewer: string;
  notes: string;
  reviewed_at: string;
};

function ReviewQueue() {
  const [reviews, setReviews] = useState<ReviewRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;

    async function loadReviews() {
      try {
        const response = await fetch(`${API_BASE_URL}/reviews`);

        if (!response.ok) {
          throw new Error("Failed to fetch review records.");
        }

        const data = await response.json();

        if (active) {
          setReviews(data.reviews ?? []);
        }
      } catch (err) {
        if (active) {
          setError(
            err instanceof Error
              ? err.message
              : "Unable to load review records.",
          );
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }

    loadReviews();

    return () => {
      active = false;
    };
  }, []);

  function formatTimestamp(seconds: number) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);

    return `${String(minutes).padStart(2, "0")}:${String(
      remainingSeconds,
    ).padStart(2, "0")}`;
  }

  return (
    <section className="panel">
      <div className="panel-heading">
        <div>
          <h3>Review Queue</h3>
          <p>Saved human review decisions from video analysis.</p>
        </div>

        <span className="neutral-pill">{reviews.length} records</span>
      </div>

      {loading && <p>Loading review records...</p>}

      {error && (
        <p role="alert" className="error-message">
          {error}
        </p>
      )}

      {!loading && !error && reviews.length === 0 && (
        <p>
          No review records found. Review a detection in the Video Library
          first.
        </p>
      )}

      {!loading && !error && reviews.length > 0 && (
        <div style={{ overflowX: "auto" }}>
          <table>
            <thead>
              <tr>
                <th>Frame</th>
                <th>Detection</th>
                <th>Confidence</th>
                <th>Timestamp</th>
                <th>Decision</th>
                <th>Reviewer</th>
                <th>Reviewed at</th>
              </tr>
            </thead>

            <tbody>
              {reviews.map((review) => (
                <tr key={review.review_id}>
                  <td>{review.frame}</td>
                  <td>{review.target_class}</td>
                  <td>{(review.confidence * 100).toFixed(1)}%</td>
                  <td>{formatTimestamp(review.timestamp_seconds)}</td>
                  <td>{review.decision.replaceAll("_", " ")}</td>
                  <td>{review.reviewer}</td>
                  <td>
                    {review.reviewed_at
                      ? new Date(review.reviewed_at).toLocaleString()
                      : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function AuditHistory() {
  const [reviews, setReviews] = useState<ReviewRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;

    async function loadAuditHistory() {
      try {
        const response = await fetch(`${API_BASE_URL}/reviews`);

        if (!response.ok) {
          throw new Error("Failed to load audit history.");
        }

        const data = await response.json();

        if (active) {
          setReviews(data.reviews ?? []);
        }
      } catch (err) {
        if (active) {
          setError(
            err instanceof Error
              ? err.message
              : "Unable to load audit history.",
          );
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }

    loadAuditHistory();

    return () => {
      active = false;
    };
  }, []);

  return (
    <section className="panel">
      <div className="panel-heading">
        <div>
          <h3>Audit History</h3>
          <p>Historical record of submitted review decisions.</p>
        </div>

        <span className="neutral-pill">{reviews.length} entries</span>
      </div>

      {loading && <p>Loading audit history...</p>}

      {error && (
        <p role="alert" className="error-message">
          {error}
        </p>
      )}

      {!loading && !error && reviews.length === 0 && (
        <p>No audit records found yet.</p>
      )}

      {!loading && !error && reviews.length > 0 && (
        <div style={{ overflowX: "auto" }}>
          <table>
            <thead>
              <tr>
                <th>Review ID</th>
                <th>Video ID</th>
                <th>Frame</th>
                <th>Decision</th>
                <th>Confidence</th>
                <th>Reviewer</th>
                <th>Date and time</th>
              </tr>
            </thead>

            <tbody>
              {reviews.map((review) => (
                <tr key={review.review_id}>
                  <td>{review.review_id}</td>
                  <td>{review.video_id}</td>
                  <td>{review.frame}</td>
                  <td>{review.decision.replaceAll("_", " ")}</td>
                  <td>{(review.confidence * 100).toFixed(1)}%</td>
                  <td>{review.reviewer}</td>
                  <td>
                    {review.reviewed_at
                      ? new Date(review.reviewed_at).toLocaleString()
                      : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function LiveMonitoring({
  selectedSafetyRule,
  configuredZone,
  onNavigate,
}: {
  selectedSafetyRule: SafetyRuleDraft | null;
  configuredZone: ZonePoint[];
  onNavigate: (page: PageId) => void;
}) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const requestRef = useRef<AbortController | null>(null);
  const timeoutRef = useRef<number | null>(null);
  const sessionIdRef = useRef(0);
  const activeRef = useRef(false);
  const startedAtRef = useRef(0);
  const [starting, setStarting] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const [latestResult, setLatestResult] = useState<LiveFrameResult | null>(
    null,
  );

  function releaseSession() {
    sessionIdRef.current += 1;
    activeRef.current = false;
    if (timeoutRef.current !== null) {
      window.clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
    requestRef.current?.abort();
    requestRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
  }

  useEffect(
    () => () => {
      sessionIdRef.current += 1;
      activeRef.current = false;
      if (timeoutRef.current !== null) {
        window.clearTimeout(timeoutRef.current);
      }
      requestRef.current?.abort();
      streamRef.current?.getTracks().forEach((track) => track.stop());
      if (videoRef.current) videoRef.current.srcObject = null;
    },
    [],
  );

  function stopMonitoring() {
    releaseSession();
    setStarting(false);
    setRunning(false);
  }

  function readableCameraError(reason: unknown) {
    if (reason instanceof DOMException) {
      if (reason.name === "NotAllowedError" || reason.name === "SecurityError") {
        return "Camera permission was denied. Allow camera access in your browser settings, then try again.";
      }
      if (reason.name === "NotFoundError" || reason.name === "DevicesNotFoundError") {
        return "No camera is available. Connect or enable a webcam and try again.";
      }
      if (reason.name === "NotReadableError" || reason.name === "TrackStartError") {
        return "The camera could not be started. It may be in use by another application.";
      }
    }
    return reason instanceof Error
      ? reason.message
      : "Unable to start webcam monitoring.";
  }

  async function captureAndEvaluate(sessionId: number) {
    if (!activeRef.current || sessionIdRef.current !== sessionId) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.videoWidth <= 0 || video.videoHeight <= 0) {
      setError("The webcam has not produced a usable frame yet.");
      stopMonitoring();
      return;
    }

    const scale = Math.min(
      640 / video.videoWidth,
      640 / video.videoHeight,
      1,
    );
    canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
    canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
    const context = canvas.getContext("2d");
    if (!context) {
      setError("This browser could not prepare a webcam frame for inference.");
      stopMonitoring();
      return;
    }
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    try {
      const blob = await new Promise<Blob | null>((resolve) =>
        canvas.toBlob(resolve, "image/jpeg", 0.75),
      );
      if (!blob) throw new Error("Unable to encode a webcam frame.");
      if (!activeRef.current || sessionIdRef.current !== sessionId) return;

      const formData = new FormData();
      formData.append("file", blob, "webcam-frame.jpg");
      formData.append("rule", JSON.stringify(selectedSafetyRule));
      if (configuredZone.length >= 3) {
        formData.append("zone", JSON.stringify(configuredZone));
      }
      formData.append(
        "timestamp_seconds",
        String((performance.now() - startedAtRef.current) / 1000),
      );

      const controller = new AbortController();
      requestRef.current = controller;
      const response = await fetch(`${API_BASE_URL}/rules/evaluate-frame`, {
        method: "POST",
        body: formData,
        signal: controller.signal,
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : JSON.stringify(data.detail || "Live frame evaluation failed."),
        );
      }
      if (!activeRef.current || sessionIdRef.current !== sessionId) return;
      setLatestResult(data as LiveFrameResult);
      setError("");
      requestRef.current = null;
      timeoutRef.current = window.setTimeout(
        () => void captureAndEvaluate(sessionId),
        1500,
      );
    } catch (reason) {
      if (
        reason instanceof DOMException &&
        reason.name === "AbortError"
      ) {
        return;
      }
      if (sessionIdRef.current !== sessionId) return;
      setError(
        reason instanceof TypeError && reason.message === "Failed to fetch"
          ? `Could not reach the Rulevia backend at ${API_BASE_URL}. Check the backend URL and CORS configuration.`
          : reason instanceof Error
            ? reason.message
            : "Live frame evaluation failed.",
      );
      stopMonitoring();
    }
  }

  async function startMonitoring() {
    if (starting || running) return;
    setError("");
    setLatestResult(null);
    if (!selectedSafetyRule) {
      setError("Choose a safety rule in Rule Builder before starting monitoring.");
      return;
    }
    if (selectedSafetyRule.condition.includes("enters_zone")) {
      setError(
        "Entry rules need adjacent observations and are not supported by single-frame live monitoring. Select a presence or overlap rule.",
      );
      return;
    }
    if (!window.isSecureContext) {
      setError(
        "Webcam access requires a secure browser context. Use localhost or HTTPS.",
      );
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("This browser does not provide webcam access.");
      return;
    }

    const sessionId = ++sessionIdRef.current;
    setStarting(true);
    try {
      const validationController = new AbortController();
      requestRef.current = validationController;
      const validationResponse = await fetch(`${API_BASE_URL}/rules/validate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          rule: selectedSafetyRule,
          zone: configuredZone.length >= 3 ? configuredZone : null,
        }),
        signal: validationController.signal,
      });
      const validation = (await validationResponse.json().catch(() => ({}))) as
        | RuleValidationResult
        | { detail?: string };
      if (sessionIdRef.current !== sessionId) return;
      requestRef.current = null;
      if (!validationResponse.ok) {
        throw new Error(
          "detail" in validation && typeof validation.detail === "string"
            ? validation.detail
            : "Rule validation failed.",
        );
      }
      if (
        !("status" in validation) ||
        validation.status !== "ready_for_evaluation"
      ) {
        const details =
          "findings" in validation
            ? validation.findings
                .filter((finding) => finding.status !== "passed")
                .map((finding) => finding.message)
                .join(" ")
            : "";
        throw new Error(
          details || "Validate and configure the selected rule before monitoring.",
        );
      }
      if (sessionIdRef.current !== sessionId) return;

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 640 },
          height: { ideal: 480 },
        },
        audio: false,
      });
      if (sessionIdRef.current !== sessionId) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      streamRef.current = stream;
      if (!videoRef.current) {
        throw new Error("Webcam preview is unavailable.");
      }
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      if (sessionIdRef.current !== sessionId) return;
      startedAtRef.current = performance.now();
      activeRef.current = true;
      setRunning(true);
      void captureAndEvaluate(sessionId);
    } catch (reason) {
      if (sessionIdRef.current === sessionId) {
        setError(
          reason instanceof TypeError && reason.message === "Failed to fetch"
            ? `Could not reach the Rulevia backend at ${API_BASE_URL}. Check the backend URL and CORS configuration.`
            : readableCameraError(reason),
        );
        setStarting(false);
        setRunning(false);
        releaseSession();
      }
    } finally {
      if (sessionIdRef.current === sessionId) setStarting(false);
    }
  }

  const zoneRequired = selectedSafetyRule
    ? ruleRequiresZone(selectedSafetyRule)
    : false;

  return (
    <section className="live-monitor-page">
      <div className="panel live-monitor-controls">
        <div>
          <span className="rule-step-label">LOCAL WEBCAM MONITORING</span>
          <h2>{selectedSafetyRule?.rule_name ?? "No rule selected"}</h2>
          <p>
            {selectedSafetyRule
              ? `${selectedSafetyRule.detection_target} · ${selectedSafetyRule.condition}`
              : "Select and validate a rule in Rule Builder to begin."}
          </p>
          {zoneRequired && (
            <p>
              {configuredZone.length >= 3
                ? `Restricted zone configured (${configuredZone.length} points).`
                : "This spatial rule requires a zone configured in Video Library."}
            </p>
          )}
        </div>
        <div className="live-monitor-actions">
          <button
            className="primary-button"
            onClick={() => void startMonitoring()}
            disabled={starting || running}
          >
            {starting ? "Starting…" : "Start Monitoring"}
          </button>
          <button
            className="secondary-button"
            onClick={stopMonitoring}
            disabled={!starting && !running}
          >
            Stop Monitoring
          </button>
          {!selectedSafetyRule && (
            <button
              className="secondary-button"
              onClick={() => onNavigate("rules")}
            >
              Open Rule Builder
            </button>
          )}
        </div>
      </div>

      {error && (
        <p className="live-monitor-error" role="alert">
          {error}
        </p>
      )}

      <div className="live-monitor-layout">
        <div className="panel live-monitor-preview">
          <div className="panel-heading">
            <h3>Webcam preview</h3>
            <span className="neutral-pill">
              {running ? "Monitoring" : starting ? "Starting" : "Stopped"}
            </span>
          </div>
          <video ref={videoRef} autoPlay playsInline muted />
          <canvas ref={canvasRef} className="live-monitor-canvas" />
          <p>
            Frames are sampled every 1.5 seconds and are not stored. Stop
            Monitoring releases the camera.
          </p>
        </div>

        <div className="panel live-monitor-results">
          <div className="panel-heading">
            <h3>Latest evaluation</h3>
            {latestResult && (
              <span className="neutral-pill">
                {latestResult.condition_met ? "Potential match" : "No match"}
              </span>
            )}
          </div>
          {!latestResult ? (
            <p>
              Start monitoring to see target detections and per-frame rule
              diagnostics.
            </p>
          ) : (
            <>
              <p>
                {latestResult.rule_name} · {latestResult.target_class} ·{" "}
                {latestResult.condition}
              </p>
              <p>
                Frame {latestResult.frame_width}×{latestResult.frame_height} ·
                {" "}session time {latestResult.timestamp_seconds.toFixed(1)}s ·{" "}
                {latestResult.detection_count} target detection(s)
              </p>
              <ul className="live-detection-list">
                {latestResult.detections.length === 0 ? (
                  <li>No target detections in this frame.</li>
                ) : (
                  latestResult.detections.map((detection, index) => (
                    <li key={`${detection.class_id}-${index}`}>
                      <strong>{detection.class_name}</strong>
                      {" "}· {(detection.confidence * 100).toFixed(1)}%
                      {" "}· {detection.matches ? "matches" : "does not match"}
                      {detection.inside_zone !== undefined &&
                        ` · bottom-center ${detection.inside_zone ? "inside" : "outside"} zone`}
                      {detection.zone_overlap_ratio !== undefined &&
                        ` · overlap ${(detection.zone_overlap_ratio * 100).toFixed(1)}%`}
                    </li>
                  ))
                )}
              </ul>
              <p>
                Potential detections require human review; live frames are
                transient and are not added to Review Queue or Audit History.
              </p>
            </>
          )}
        </div>
      </div>
    </section>
  );
}

const featureDetails: Record<
  Exclude<PageId, "overview">,
  {
    icon: string;
    heading: string;
    description: string;
    next: string;
    target: PageId;
  }
> = {
  live: {
    icon: "◉",
    heading: "Live monitoring is not connected yet",
    description:
      "This workspace will display camera streams, detection overlays, and potential safety events after the live monitoring pipeline is implemented.",
    next: "Explore safety rules",
    target: "rules",
  },
  rules: {
    icon: "⌘",
    heading: "Your rule builder is coming together",
    description:
      "Describe a visual safety requirement in plain language. Rulevia will turn it into a structured request and check it against registered detector and evaluation capabilities.",
    next: "Explore the video library",
    target: "videos",
  },
  videos: {
    icon: "▣",
    heading: "Your video library is ready for integration",
    description:
      "The upload workflow will let you select recorded CCTV footage, validate its format, and prepare it for frame extraction and analysis.",
    next: "View analysis results",
    target: "analysis",
  },
  analysis: {
    icon: "⌁",
    heading: "Analysis results will appear here",
    description:
      "Processed videos will show timestamped detections, evidence frames, rule outcomes, and confidence information. Potential events will remain subject to human review.",
    next: "Explore the review queue",
    target: "review",
  },
  backtesting: {
    icon: "↻",
    heading: "Evaluate rules against historical footage",
    description:
      "Backtesting will replay selected rules against recorded videos and summarize potential events, missed detections, and evaluation metrics when labelled reference data is available.",
    next: "Open the rule builder",
    target: "rules",
  },
  review: {
    icon: "☑",
    heading: "No events awaiting review",
    description:
      "Potential safety events will appear here with evidence frames and timestamps. Reviewers will be able to confirm, dismiss, or mark uncertain events for further investigation.",
    next: "Explore analysis results",
    target: "analysis",
  },
  audit: {
    icon: "≡",
    heading: "Your audit history will appear here",
    description:
      "This page will track rule changes, analysis runs, review decisions, and other important actions to support traceability.",
    next: "Explore workspace settings",
    target: "settings",
  },
  settings: {
    icon: "⚙",
    heading: "Workspace settings",
    description:
      "Configuration for AI model access, local storage, and future deployment options will be added as those integrations are implemented.",
    next: "Return to overview",
    target: "overview",
  },
};

function FeaturePage({
  page,
  onNavigate,
  selectedSafetyRule,
  onSelectSafetyRule,
  videoResult,
  onVideoResultChange,
  configuredZone,
  onConfiguredZoneChange,
}: {
  page: Exclude<PageId, "overview">;
  onNavigate: (page: PageId) => void;
  selectedSafetyRule: SafetyRuleDraft | null;
  onSelectSafetyRule: (rule: SafetyRuleDraft | null) => void;
  videoResult: VideoUploadResult | null;
  onVideoResultChange: (result: VideoUploadResult | null) => void;
  configuredZone: ZonePoint[];
  onConfiguredZoneChange: (zone: ZonePoint[]) => void;
}) {
  if (page === "live") {
    return (
      <LiveMonitoring
        selectedSafetyRule={selectedSafetyRule}
        configuredZone={configuredZone}
        onNavigate={onNavigate}
      />
    );
  }

  if (page === "rules") {
    return (
      <RuleBuilder
        onNavigate={onNavigate}
        selectedSafetyRule={selectedSafetyRule}
        onSelectSafetyRule={onSelectSafetyRule}
        configuredZone={configuredZone}
      />
    );
  }

  if (page === "videos") {
    return (
      <VideoLibrary
        selectedSafetyRule={selectedSafetyRule}
        result={videoResult}
        onVideoResultChange={onVideoResultChange}
        configuredZone={configuredZone}
        onConfiguredZoneChange={onConfiguredZoneChange}
      />
    );
  }

  if (page === "review") {
    return <ReviewQueue />;
  }
  if (page === "audit") {
    return <AuditHistory />;
  }
  const feature = featureDetails[page];

  return (
    <section className="feature-page">
      <div className="feature-hero">
        <div className="feature-symbol">{feature.icon}</div>
        <span className="coming-soon-label">IMPLEMENTATION IN PROGRESS</span>
        <h2>{feature.heading}</h2>
        <p>{feature.description}</p>
        <div className="feature-actions">
          <button
            className="primary-button"
            onClick={() => onNavigate(feature.target)}
          >
            {feature.next} <span>→</span>
          </button>
          <button
            className="secondary-button"
            onClick={() => onNavigate("overview")}
          >
            Back to overview
          </button>
        </div>
      </div>

      <div className="feature-info-grid">
        <div className="feature-info-card">
          <span className="info-number">01</span>
          <h3>Grounded in evidence</h3>
          <p>
            Show the footage and timestamp associated with a potential event.
          </p>
        </div>
        <div className="feature-info-card">
          <span className="info-number">02</span>
          <h3>Deterministic evaluation</h3>
          <p>Evaluate validated rules against structured detection results.</p>
        </div>
        <div className="feature-info-card">
          <span className="info-number">03</span>
          <h3>Human oversight</h3>
          <p>
            Keep uncertain detections reviewable instead of treating them as
            facts.
          </p>
        </div>
      </div>
    </section>
  );
}

export default App;
