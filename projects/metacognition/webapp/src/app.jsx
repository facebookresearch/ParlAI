import React from "react";
import ReactDOM from "react-dom";
import { BaseFrontend, LoadingScreen } from "./components/core_components.jsx";
import { useMephistoTask } from "mephisto-task";

/* ================= Application Components ================= */

function MainApp() {
  const {
    blockedReason,
    blockedExplanation,
    isPreview,
    isLoading,
    initialTaskData,
    handleSubmit,
    isOnboarding,
  } = useMephistoTask();

  if (blockedReason !== null) {
    return (
      <section className="hero is-medium is-danger">
        <div class="hero-body">
          <h2 className="title is-3">{blockedExplanation}</h2>{" "}
        </div>
      </section>
    );
  }
  if (isLoading) {
    return <LoadingScreen />;
  }
  if (isPreview) {
    return (
      <section className="hero is-medium is-link">
        <div class="hero-body">
          <div className="title is-3">
            Can our chatbot answer questions correctly 💯 and confidently 🙋 or does just spin us some yarn 🧶?
          </div>
          <div className="subtitle is-4">
            You'll be asked to judge <b>correctness</b> and <b>certainty</b> of chatbot answers to questions---don't worry, you will be shown some correct answers, so you need not know the answers yourself.
          </div>
        </div>
      </section>
    );
  }

  return (
    <div>
      <BaseFrontend
        taskData={initialTaskData}
        onSubmit={handleSubmit}
        isOnboarding={isOnboarding}
      />
    </div>
  );
}

ReactDOM.render(<MainApp />, document.getElementById("app"));
