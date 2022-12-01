import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Mainpage from './pages/MainPage';
import SidePage from './pages/SidePage';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Mainpage />} />
        <Route path="/side" element={<SidePage />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
