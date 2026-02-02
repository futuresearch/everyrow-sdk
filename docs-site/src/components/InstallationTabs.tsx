"use client";

import { useState, createContext, useContext, ReactNode } from "react";

type Agent = "python-sdk" | "claude-code" | "claude-desktop" | "codex" | "gemini" | "cursor";
type IntegrationType = "sdk" | "skills" | "mcp" | "plugin";

interface TabContextValue {
  selectedAgent: Agent;
  selectedType: IntegrationType | null;
  isActive: (agent: Agent, type: IntegrationType) => boolean;
}

const TabContext = createContext<TabContextValue | null>(null);

const AGENTS: { id: Agent; label: string }[] = [
  { id: "python-sdk", label: "Python SDK" },
  { id: "claude-code", label: "Claude Code" },
  { id: "claude-desktop", label: "Claude Desktop" },
  { id: "codex", label: "Codex" },
  { id: "gemini", label: "Gemini" },
  { id: "cursor", label: "Cursor" },
];

const TYPES: { id: IntegrationType; label: string }[] = [
  { id: "skills", label: "Skills" },
  { id: "mcp", label: "MCP" },
  { id: "plugin", label: "Plugin" },
];

// Which integration types are available for each agent (null means no method selector)
const AGENT_TYPES: Record<Agent, IntegrationType[] | null> = {
  "python-sdk": null,
  "claude-code": ["skills", "mcp", "plugin"],
  "claude-desktop": ["mcp"],
  "codex": ["skills"],
  "gemini": ["skills", "plugin"],
  "cursor": ["skills"],
};

interface InstallationTabsProps {
  children: ReactNode;
}

export function InstallationTabs({ children }: InstallationTabsProps) {
  const [selectedAgent, setSelectedAgent] = useState<Agent>("python-sdk");
  const [selectedType, setSelectedType] = useState<IntegrationType | null>(null);

  // Get available types for selected agent (null means no method selector needed)
  const availableTypes = AGENT_TYPES[selectedAgent];

  // Determine effective type
  const effectiveType = availableTypes === null
    ? null
    : availableTypes.includes(selectedType as IntegrationType)
      ? selectedType
      : availableTypes[0];

  const handleAgentChange = (agent: Agent) => {
    setSelectedAgent(agent);
    const newAvailableTypes = AGENT_TYPES[agent];
    // Auto-select first available type if current isn't valid
    if (newAvailableTypes === null) {
      setSelectedType(null);
    } else if (!newAvailableTypes.includes(selectedType as IntegrationType)) {
      setSelectedType(newAvailableTypes[0]);
    }
  };

  const isActive = (agent: Agent, type: IntegrationType) => {
    // For agents with no method selector, match on agent only with "sdk" type
    if (AGENT_TYPES[agent] === null) {
      return selectedAgent === agent && type === "sdk";
    }
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
          {availableTypes !== null && (
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
          )}
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

  // For agents with no method selector, just show the agent name
  const hasMethodSelector = AGENT_TYPES[agent] !== null;
  const heading = hasMethodSelector ? `${agentLabel} with ${typeLabel}` : agentLabel;

  return (
    <div
      className={`tab-content ${isActive ? "active" : ""}`}
      data-agent={agent}
      data-type={type}
    >
      <h3 className="tab-content-heading">{heading}</h3>
      {children}
    </div>
  );
}
