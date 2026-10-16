import { useEffect, useState } from "react";
import "./App.css";

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
  const testSafetyRule: SafetyRuleDraft = {
    rule_name: "Person Detection Check",
    description: "Check whether a person is present in the recorded video.",
    hazard_type: "other",
    detection_target: "person",
    condition: "present",
    severity: "medium",
    required_evidence: ["person_detection", "timestamp", "bounding_box"],
    recommended_action: "Review the detected person.",
    confidence_threshold: 0.25,
    limitations: ["Person detection does not prove a safety violation."],
  };

  useEffect(() => {
    let active = true;

    async function checkBackend() {
      try {
        const response = await fetch("http://127.0.0.1:8000/health");
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
              testSafetyRule={testSafetyRule}
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
  condition: string;
  severity: string;
  required_evidence: string[];
  recommended_action: string;
  confidence_threshold: number;
  limitations: string[];
};

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

const exampleInstructions = [
  "Alert when a person is detected without a hardhat.",
  "Flag people entering a restricted area.",
  "Identify potential falls that require human review.",
];

function RuleBuilder({
  onNavigate,
  onSelectSafetyRule,
}: {
  onNavigate: (page: PageId) => void;
  onSelectSafetyRule: (rule: SafetyRuleDraft | null) => void;
}) {
  const [instruction, setInstruction] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [rule, setRule] = useState<SafetyRuleDraft | null>(null);

  const [validation, setValidation] = useState<RuleValidationResult | null>(
    null,
  );
  const [validating, setValidating] = useState(false);

  async function generateRule() {
    setLoading(true);
    setError("");
    setRule(null);

    try {
      const response = await fetch("http://127.0.0.1:8000/rules/generate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ instruction }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Unable to generate a safety rule.");
      }

      setRule(data.rule);
      onSelectSafetyRule(data.rule);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Something went wrong. Check that the backend is running.",
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
      const response = await fetch("http://127.0.0.1:8000/rules/validate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rule }),
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
          onChange={(event) => setInstruction(event.target.value)}
          placeholder="Example: Alert me when a worker is detected without a hardhat in the manufacturing area..."
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
                }}
              >
                {example}
              </button>
            ))}
          </div>
        </div>

        {error && (
          <div className="rule-error" role="alert">
            <strong>Generation failed</strong>
            <p>{error}</p>
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
              <span>Detection target</span>
              <strong>{rule.detection_target}</strong>
            </div>
            <div className="generated-field">
              <span>Proposed confidence threshold</span>
              <strong>{(rule.confidence_threshold * 100).toFixed(0)}%</strong>
            </div>
          </div>

          <div className="generated-section">
            <h3>Trigger condition</h3>
            <p>{rule.condition}</p>
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

function VideoLibrary({
  selectedSafetyRule,
}: {
  selectedSafetyRule: SafetyRuleDraft | null;
}) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<VideoUploadResult | null>(null);

  const [reviewingFrame, setReviewingFrame] = useState("");
  const [reviewMessage, setReviewMessage] = useState("");
  const [reviewError, setReviewError] = useState("");
  const [reviewMessageFrame, setReviewMessageFrame] = useState("");

  const [evaluating, setEvaluating] = useState(false);
  const [evaluationError, setEvaluationError] = useState("");
  const [evaluationResult, setEvaluationResult] = useState<any>(null);

  async function uploadVideo() {
    if (!selectedFile) {
      setError("Please select a video first.");
      return;
    }

    setUploading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch(
        "http://127.0.0.1:8000/videos/upload-extract",
        {
          method: "POST",
          body: formData,
        },
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Video upload failed.");
      }

      setResult(data as VideoUploadResult);
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
    frame: any,
    match: any,
    decision: "confirmed" | "dismissed" | "needs_further_review",
  ) {
    const reviewKey = `${frame.frame}-${decision}`;
    setReviewMessageFrame(frame.frame);

    setReviewingFrame(reviewKey);
    setReviewMessage("");
    setReviewError("");

    try {
      const response = await fetch("http://127.0.0.1:8000/reviews", {
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
        "http://127.0.0.1:8000/rules/evaluate-video",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            video_id: result.video_id,
            rule,
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

      setEvaluationResult(data);
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
            setResult(null);
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
                    src={`http://127.0.0.1:8000${frame.image_url}`}
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

          <section className="panel evaluation-panel">
            <div className="panel-heading">
              <div>
                <h3>Safety Rule Evaluation</h3>
                <p>Check whether a person is detected in this video.</p>
              </div>
            </div>

            <button
              type="button"
              className="primary-button"
              onClick={evaluateSafetyRule}
              disabled={evaluating}
            >
              {evaluating ? "Evaluating video..." : "Evaluate Safety Rule"}
            </button>

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
                  <strong>Review status:</strong>{" "}
                  {evaluationResult.evaluation_status}
                </p>

                <h4>Matching Frames</h4>

                {evaluationResult.matching_frames?.length > 0 ? (
                  <div className="extracted-frames-grid">
                    {evaluationResult.matching_frames.map((frame: any) => (
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
                            src={`http://127.0.0.1:8000/frames/${result.video_id}/${frame.frame}`}
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

                          {frame.matches
                            .filter(
                              (match: any, index: number, matches: any[]) =>
                                index ===
                                matches.findIndex(
                                  (item: any) =>
                                    item.class_name === match.class_name,
                                ),
                            )
                            .map((match: any, index: number) => {
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

                          {frame.matches.map((match: any, index: number) => (
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
        const response = await fetch("http://127.0.0.1:8000/reviews");

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
        const response = await fetch("http://127.0.0.1:8000/reviews");

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
      "You will describe requirements such as “Alert me when a person enters the restricted zone without a hardhat.” Rulevia will turn the instruction into a structured rule for validation.",
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
  testSafetyRule,
}: {
  page: Exclude<PageId, "overview">;
  onNavigate: (page: PageId) => void;
  selectedSafetyRule: SafetyRuleDraft | null;
  onSelectSafetyRule: (rule: SafetyRuleDraft | null) => void;
  testSafetyRule: SafetyRuleDraft;
}) {
  // Show the real Rule Builder instead of the placeholder.
  if (page === "rules") {
    return (
      <RuleBuilder
        onNavigate={onNavigate}
        onSelectSafetyRule={onSelectSafetyRule}
      />
    );
  }

  if (page === "videos") {
    return (
      <VideoLibrary selectedSafetyRule={selectedSafetyRule ?? testSafetyRule} />
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
