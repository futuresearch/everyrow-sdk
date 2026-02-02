"use client";

import { useState, createContext, useContext, ReactNode } from "react";

type Agent = "claude-code" | "claude-desktop" | "codex" | "gemini" | "cursor" | "other";
type IntegrationType = "skills" | "mcp" | "sdk";

interface TabContextValue {
  selectedAgent: Agent;
  selectedType: IntegrationType;
  isActive: (agent: Agent, type: IntegrationType) => boolean;
}

const TabContext = createContext<TabContextValue | null>(null);

const AGENTS: { id: Agent; label: string }[] = [
  { id: "claude-code", label: "Claude Code" },
  { id: "claude-desktop", label: "Claude Desktop" },
  { id: "codex", label: "Codex" },
  { id: "gemini", label: "Gemini" },
  { id: "cursor", label: "Cursor" },
  { id: "other", label: "Other" },
];

const TYPES: { id: IntegrationType; label: string }[] = [
  { id: "skills", label: "Skills" },
  { id: "mcp", label: "MCP" },
  { id: "sdk", label: "Python SDK" },
];

// Which integration types are available for each agent
const AGENT_TYPES: Record<Agent, IntegrationType[]> = {
  "claude-code": ["skills", "mcp"],
  "claude-desktop": ["mcp"],
  "codex": ["skills"],
  "gemini": ["skills"],
  "cursor": ["skills"],
  "other": ["sdk"],
};

interface InstallationTabsProps {
  children: ReactNode;
}

export function InstallationTabs({ children }: InstallationTabsProps) {
  const [selectedAgent, setSelectedAgent] = useState<Agent>("claude-code");
  const [selectedType, setSelectedType] = useState<IntegrationType>("skills");

  // Get available types for selected agent
  const availableTypes = AGENT_TYPES[selectedAgent];

  // If current type isn't available for new agent, switch to first available
  const effectiveType = availableTypes.includes(selectedType)
    ? selectedType
    : availableTypes[0];

  const handleAgentChange = (agent: Agent) => {
    setSelectedAgent(agent);
    // Auto-select first available type if current isn't valid
    if (!AGENT_TYPES[agent].includes(selectedType)) {
      setSelectedType(AGENT_TYPES[agent][0]);
    }
  };

  const isActive = (agent: Agent, type: IntegrationType) => {
    return selectedAgent === agent && effectiveType === type;
  };

  return (
    <TabContext.Provider value={{ selectedAgent, selectedType: effectiveType, isActive }}>
      <div className="installation-tabs">
        <div className="tab-selectors">
          <div className="tab-selector-row">
            <span className="tab-selector-label">Platform</span>
            <div className="tab-selector-options">
              {AGENTS.map((agent) => (
                <button
                  key={agent.id}
                  className={`tab-option ${selectedAgent === agent.id ? "active" : ""}`}
                  onClick={() => handleAgentChange(agent.id)}
                >
                  {agent.label}
                </button>
              ))}
            </div>
          </div>
          <div className="tab-selector-row">
            <span className="tab-selector-label">Method</span>
            <div className="tab-selector-options">
              {TYPES.map((type) => {
                const isAvailable = availableTypes.includes(type.id);
                return (
                  <button
                    key={type.id}
                    className={`tab-option ${effectiveType === type.id ? "active" : ""} ${!isAvailable ? "disabled" : ""}`}
                    onClick={() => isAvailable && setSelectedType(type.id)}
                    disabled={!isAvailable}
                    title={!isAvailable ? `${type.label} not available for ${AGENTS.find(a => a.id === selectedAgent)?.label}` : undefined}
                  >
                    {type.label}
                  </button>
                );
              })}
            </div>
          </div>
        </div>
        <div className="tab-contents">
          {children}
        </div>
      </div>
    </TabContext.Provider>
  );
}

interface TabContentProps {
  agent: Agent;
  type: IntegrationType;
  children: ReactNode;
}

export function TabContent({ agent, type, children }: TabContentProps) {
  const context = useContext(TabContext);

  // During SSR or if no context, show all content (for no-JS readers)
  const isActive = context?.isActive(agent, type) ?? true;

  // Get readable labels for the heading
  const agentLabel = AGENTS.find(a => a.id === agent)?.label ?? agent;
  const typeLabel = TYPES.find(t => t.id === type)?.label ?? type;

  return (
    <div
      className={`tab-content ${isActive ? "active" : ""}`}
      data-agent={agent}
      data-type={type}
    >
      <h3 className="tab-content-heading">{agentLabel} with {typeLabel}</h3>
      {children}
    </div>
  );
}
