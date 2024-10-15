import React, { useEffect, useState } from 'react';

function App() {
  const [number, setNumber] = useState(null);

  useEffect(() => {
    fetch('http://localhost:8000/data')
      .then(response => response.json())
      .then(data => setNumber(data.number))
      .catch(error => console.error('Error fetching data:', error));
  }, []);

  return (
    <div className="App">
      <h1>Random Number from Backend</h1>
      {number !== null ? <p>{number}</p> : <p>Loading...</p>}
    </div>
  );
}

export default App;
