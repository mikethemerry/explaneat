import React from "react";
import { Link, useLocation } from "react-router-dom";

const NavLink: React.FC<{ to: string; children: React.ReactNode }> = ({
  to,
  children,
}) => {
  const location = useLocation();
  const isActive = location.pathname === to;
  return (
    <Link
      to={to}
      className={`px-3 py-2 rounded-md text-sm font-medium ${
        isActive
          ? "bg-primary-700 text-white"
          : "text-neutral-300 hover:bg-primary-700 hover:text-white"
      }`}
    >
      {children}
    </Link>
  );
};

const Layout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return (
    <div className="min-h-screen bg-neutral-100">
      <nav className="bg-primary-600">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <span className="text-white text-xl font-bold">
                  NEAT Visualizer
                </span>
              </div>
              <div className="hidden md:block">
                <div className="ml-10 flex items-baseline space-x-4">
                  <NavLink to="/">Dashboard</NavLink>
                  <NavLink to="/create">Create Model</NavLink>
                </div>
              </div>
            </div>
          </div>
        </div>
      </nav>

      <main>
        {children}
        {/* <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">{children}</div> */}
      </main>
    </div>
  );
};

export default Layout;
