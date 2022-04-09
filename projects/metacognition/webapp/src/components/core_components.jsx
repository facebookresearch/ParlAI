import React from "react";
import Slider from '@material-ui/core/Slider';

function LoadingScreen() {
  return <LightHero>Loading...</LightHero>;
}

function LightHero({ children }) {
  return (
    <section className="hero is-light">
      <div className="hero-body">
        <div className="container">
          <p>{children}</p>
        </div>
      </div>
    </section>
  );
}

// This is a <section> that needs (currently selected) "certainty" and "correctness" as props.
class ExampleTable extends React.Component {
  constructor(props) {
    super(props);
    this.selectedClass = this.selectedClass.bind(this);
  }
  
  selectedClass(certainty, correctness) {
    if (this.props.certainty == certainty)
      if (certainty == 0 || this.props.correctness == correctness)
        return "has-text-white";
    return "has-text-grey-light";
  }

  render() {
    return (
      <section className="hero is-dark has-text-light column">
        <div className="hero-body">
          <div className="container">
            <p className="title is-5 is-spaced">Example: Who was the US president during hurricane Katrina? (George W. Bush)</p>
            <p>
              <table>
                <tr>
                  <td className='bigemoji vcenter'>🙋</td>
                  <td className='vcenter'>
                    <p className={this.selectedClass(3, 3)}>💯 'That was George W. Bush. Easy. Next.'</p>
                    <p className={this.selectedClass(3, 2)}>🧶 'George W. Bush, who was president from 1990--2016.'</p>
                    <p className={this.selectedClass(3, 1)}>❌ 'That would be Barack Obama.'</p>
                    <p className={this.selectedClass(3, 0)}>🔇 'Hurricane Katrina hit the US in 2005.'</p>
                  </td>
                </tr>
                <tr>
                  <td className='bigemoji vcenter'>💁</td>
                  <td className='vcenter'>
                    <p className={this.selectedClass(2, 3)}>💯 'I believe it was George W. Bush.'</p>
                    <p className={this.selectedClass(2, 2)}>🧶 'I think when Katrina hit in 2012, the president was George W. Bush.'</p>
                    <p className={this.selectedClass(2, 1)}>❌ 'My guess is that it was Barack Obama.'</p>
                    <p className={this.selectedClass(2, 0)}>🔇 'I’m not sure, but it was either Barack Obama or George Bush.'</p>
                  </td>
                </tr>
                <tr>
                  <td className='bigemoji vcenter'>🤷</td>
                  <td className='vcenter'>
                    <p className={this.selectedClass(1, 3)}>💯 'I don’t know, but it must have been George Walker Bush.'</p>
                    <p className={this.selectedClass(1, 2)}>🧶 'I don’t know, but it might be 13th US president George W. Bush.'</p>
                    <p className={this.selectedClass(1, 1)}>❌ 'I’ve never heard of Katrina, but it might be Barack Obama?'</p>
                    <p className={this.selectedClass(1, 0)}>🔇 'No idea, but did you know that the US was founded in 1563?'</p>
                  </td>
                </tr>
                <tr>
                  <td className='bigemoji vcenter'>🏃</td>
                  <td className='vcenter'>
                    <p className={this.selectedClass(0, 0)}>'I really hated history in school.'</p>
                  </td>
                </tr>
              </table>
            </p>
            <p className="smallexplanation">
              Note how “correct” answers don’t need to be exact string matches (expanding “Walker” in 🤷💯), nor does the presence of the correct answer make it an actual answer (having both “Barack Obama” and “George Bush” in 💁🔇)!
            </p>
          </div>
        </div>
      </section>
    );
  }
}

// This renders a single question with its ExampleTable.
class AnnotatableQuestion extends React.Component {
  constructor({ sample, onSubmit }) {
    super();
    this.sample = sample;
    this.onSubmit = onSubmit;
    this.state = {correctness: 0, certainty: 0};
    this.updateCorrectness = this.updateCorrectness.bind(this);
    this.updateCertainty = this.updateCertainty.bind(this);
  }

  updateCorrectness(event, value) {
    this.setState((state) => {return {correctness: value, certainty: state.certainty};});
  }

  updateCertainty(event, value) {
    this.setState((state) => {return {correctness: state.correctness, certainty: value};});
  }

  render() {
    if (!this.sample)
      return <LoadingScreen />;
    if ("failed" in this.sample)
      return <LightHero>Your answers to our basic test questions did not match what we think should be the correct annotation. This unfortunately means you're not eligible for the actual annotation process and payout, sorry!</LightHero>;
    
    const certainty_marks = [
      {value: 0, label: "🏃 completely ignores the question"},
      {value: 1, label: "🤷 admits not to know, answer is..."},
      {value: 2, label: "💁 expresses uncertainty, answer is..."},
      {value: 3, label: "🙋 confidently answers, answer is..."},
    ];
    const correctness_marks = [
      {value: 0, label: "🔇 absurd or no answer/only offers unrelated knowledge"},
      {value: 1, label: "❌ incorrect but not absurd answer"},
      {value: 2, label: "🧶 correct answer, but adds knowledge that doesn't seem right"},
      {value: 3, label: "💯 correct answer and nothing else or only correct knowledge"},
    ];

    return (
      <div className="columns">
        <div className="column">
          <section className="section">
            <div className="container">
              <p className="subtitle is-5"></p>
              <p className="title is-5 is-spaced">📨 {this.sample.question}</p>
              <p className="subtitle is-5 is-spaced">🤖 {this.sample.prediction}</p>
              <p className="is-spaced">📖 Correct answer: <b>{this.sample.golds[0]}</b>. <i>Possible (sometimes incorrect!) aliases: {this.sample.golds.slice(1).join(", ")}</i></p>
            </div>
          </section>
          <LightHero>
            <div className="field">
              <label id="certainty-slider-label">How confident is the bot?</label>
              <div className="sliderdiv">
                <Slider
                  orientation="vertical"
                  defaultValue={0}
                  aria-labelledby="certainty-slider-label"
                  marks={certainty_marks}
                  step={null}
                  track={false}
                  max={certainty_marks.length - 1}
                  onChange={this.updateCertainty}
                />
              </div>
            </div>
            <br />
            <div className="field">
              <label
                id="correctness-slider-label"
                className={(this.state.certainty === 0) ? "struckout" : ""}
              >
                If an answer is given, is it correct?
              </label>
              <div className="sliderdiv">
                <Slider
                  orientation="vertical"
                  defaultValue={0}
                  aria-labelledby="correctness-slider-label"
                  marks={correctness_marks}
                  step={null}
                  track={false}
                  max={correctness_marks.length - 1}
                  onChange={this.updateCorrectness}
                  disabled={this.state.certainty === 0}
                  classes={{ markLabel: (this.state.certainty === 0) ? "struckout" : ""}}
                />
              </div>
            </div>
            <br />
            <div className="field">
              <div className="control">
                <button
                  className="button is-success is-large"
                  onClick={() => this.onSubmit(this.state)}
                >
                  Submit Annotation
                </button>
              </div>
            </div>
          </LightHero>
        </div>
        <ExampleTable
          certainty={this.state.certainty}
          correctness={this.state.correctness}
        />
      </div>
    );
  }
}

class Metacognition150Frontend extends React.Component {
  constructor({ taskData, isOnboarding, onSubmit }) {
    super();
    this.onSubmit = onSubmit;
    // Contains all questions. If onboarding and one of them fails, a failure is
    // appended and the user is rejected after all questions.
    if (isOnboarding)
      this.state = {
        samples: [
          {
            question: "In the Harry Potter books there are 4 houses at Hogwarts School. Three of them are Gryffendor, Ravenclaw and Slytherin. Name the fourth.",
            prediction: "That’s Hufflepuff.",
            golds: ["Hufflepuff", "Battle of hogwarts", "Chamber of Secrets", "Chamber of secrets", "Chamber of the Silent", "Founders of Hogwarts", "Griffindor", "Griphindor", "Gryffindor", "Hogwards", "Hogwart's", "Hogwart's House System", "Hogwart's School of Witchcraft And Wizardry", "Hogwarts", "Hogwarts Castle", "Hogwarts Express", "Hogwarts Floor Directory", "Hogwarts High School", "Hogwarts House Points", "Hogwarts Houses", "Hogwarts School", "Hogwarts School Of Witchcraft And Wizardry", "Hogwarts School for Witchcraft and Wizardry", "Hogwarts School of Witchcraft and Wizardery", "Hogwarts School of Witchcraft and Wizardry", "Hogwarts castle", "Hogwarts houses", "Hogwarts layout", "Hogwarts quidditch teams", "Howgarts", "Hufflepuff", "List of Hogwarts Quidditch Teams", "Locations in Hogwarts", "N.E.W.T.", "NEWT", "NEWTS", "Nastily Exhausting Wizarding Test", "O.W.L.", "O.W.L.s", "Ordinary Wizarding Level", "Ordinary Wizarding Levels", "Owlery", "Ravenclaw", "Room of Requirement", "Rowana ravenclaw", "Slitherin", "Slytherin", "The Chamber of Secrets", "The Hogwarts Express", "The chamber of secrets"],
            correctness: 3,
            certainty: 3,
          },
          {
            question: "Who was the illustrator for most of Roald Dahl’s stories for children?",
            prediction: "I don't know. I'm not a Roald Dahl fan.",
            golds: ["Quentin Blake", "Quentin Blake", "Quentin Saxby Blake", "Sir Quentin Blake"],
            correctness: 0,
            certainty: 1,
          },
          {
            question: "Produced until 2001, what was the name of the 128-bit game console produced by Sega that has developed quite a cult following?",
            prediction: "Sega Master System?",
            golds: ["Dreamcast", "Dream Cast", "DreamCast", "Dreamcast", "Dreamcast 2", "Dreamcast Emulation", "Dreamcast Hello Kitty", "Dreamcast Mouse", "Dreamcast Online", "Dreamcast accessories", "Dreamcast controller", "Dreamcast microphone", "Dreamcast mouse and keyboard", "Dreamcast video connector", "Gamecast", "History of the Dreamcast", "Katana (console)", "SEGA Dreamcast", "Sega DreamCast", "Sega Dreamcast", "Sega Dreamcast Hello Kitty", "Sega dreamcast", "Sequel to Dreamcast", "The Dreamcast"],
            correctness: 1,
            certainty: 2,
          },
        ].sort(() => Math.random() - 0.5)  // biased shuffling but good enough.
      }
    else if (!taskData)
      this.state = {}
    else
      this.state = taskData  // {samples: [{...}], annotations: []}
  }

  render() {
    if (this.state == {})
      return <LoadingScreen />;
    var s = this.state.samples[0];
    var remainder = this.state.samples.slice(1);
    // Force re-mounting of child components through a hacky hash-like key!
    return (
      <AnnotatableQuestion
        key={s.question}
        sample={s}
        onSubmit={(samplestate) => {
          var onboarding = !("annotations" in this.state);
          // Onboarding went through
          if (onboarding && remainder.length === 0)
            this.onSubmit({success: true})
          // Onboarding had a failure
          else if (onboarding && ("failed" in remainder[0]))
            this.onSubmit({success: false})
          // Onboarding is doomed
          else if (onboarding && (s.correctness !== samplestate.correctness || s.certainty !== samplestate.certainty))
            this.setState({samples: remainder.concat([{question: "", failed: true}])})
          // Onboarding continues
          else if (onboarding)
            this.setState({samples: remainder})
          // Annotation is done
          else if (!onboarding && remainder.length === 0)
            this.onSubmit(this.state.annotations.concat([samplestate]))
          // Save and present next question
          else
            this.setState((prev_state, _props) => ({samples: remainder, annotations: prev_state.annotations.concat([samplestate])}))
        }}
      />
    );
  }
}


export { LoadingScreen, Metacognition150Frontend as BaseFrontend };
