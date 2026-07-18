import { useState } from "preact/hooks";

export default function App() {
  const [count, setCount] = useState(0);

  return (
    <div>
      <h1>Counter!</h1>
      <h2>{count}</h2>
      <button
        onClick={() => {
          setCount(count + 1);
        }}
      >
        Increment
      </button>
      <button
        onClick={() => {
          setCount(count - 1);
        }}
      >
        Decrement
      </button>
    </div>
  );
}