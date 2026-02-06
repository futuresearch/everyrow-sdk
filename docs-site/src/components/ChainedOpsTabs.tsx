"use client";

import { useState, createContext, useContext, ReactNode } from "react";

type Step = "screen" | "dedupe" | "merge" | "rank" | "research";

interface StepperContextValue {
  activeStep: Step;
  setActiveStep: (step: Step) => void;
}

const StepperContext = createContext<StepperContextValue | null>(null);

const STEPS: { id: Step; label: string }[] = [
  { id: "screen", label: "Screen" },
  { id: "dedupe", label: "Dedupe" },
  { id: "merge", label: "Merge" },
  { id: "rank", label: "Rank" },
  { id: "research", label: "Research" },
];

interface ChainedOpsTabsProps {
  children: ReactNode;
}

export function ChainedOpsTabs({ children }: ChainedOpsTabsProps) {
  const [activeStep, setActiveStep] = useState<Step>("screen");

  return (
    <StepperContext.Provider value={{ activeStep, setActiveStep }}>
      <div className="chained-ops-tabs">
        <div className="chained-ops-stepper">
          {STEPS.map((step, index) => (
            <button
              key={step.id}
              className={`chained-ops-step ${activeStep === step.id ? "active" : ""}`}
              onClick={() => setActiveStep(step.id)}
              data-step={index + 1}
            >
              <span className="chained-ops-step-label">{step.label}</span>
            </button>
          ))}
        </div>
        <div className="chained-ops-content">
          {children}
        </div>
      </div>
    </StepperContext.Provider>
  );
}

interface StepContentProps {
  step: Step;
  children: ReactNode;
}

export function StepContent({ step, children }: StepContentProps) {
  const context = useContext(StepperContext);
  const isActive = context?.activeStep === step;

  if (!isActive) return null;

  return (
    <div className="chained-ops-step-content" data-step={step}>
      {children}
    </div>
  );
}
