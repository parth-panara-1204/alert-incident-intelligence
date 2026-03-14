import React, { useState } from "react";

const VALID_USERNAME = "admin";
const VALID_PASSWORD = "1234";

export default function Login({ onLogin }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  const submit = (event) => {
    event.preventDefault();

    if (username === VALID_USERNAME && password === VALID_PASSWORD) {
      setError("");
      onLogin();
      return;
    }

    setError("Invalid username or password");
  };

  return (
    <div className="login-container">
      <h1>Alert Incident Intelligence</h1>
      <form onSubmit={submit} className="login-form">
        <label>
          Username
          <input
            value={username}
            onChange={(event) => setUsername(event.target.value)}
            autoComplete="username"
          />
        </label>
        <label>
          Password
          <input
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            autoComplete="current-password"
          />
        </label>
        <button type="submit">Sign In</button>
        {error && <div className="error">{error}</div>}
      </form>
    </div>
  );
}
