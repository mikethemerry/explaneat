import React from "react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import ModelList from "./components/ModelList";
import ModelDetail from "./components/ModelDetail";
import ModelForm from "./components/ModelForm";

const App: React.FC = () => {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<ModelList />} />
          <Route path="/model/:id" element={<ModelDetail />} />
          <Route path="/create" element={<ModelForm mode="create" />} />
          <Route path="/edit/:id" element={<ModelForm mode="edit" />} />
        </Routes>
      </Layout>
    </Router>
  );
};

export default App;
