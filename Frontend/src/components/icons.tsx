import React from 'react';

export const Logo = (props: React.HTMLAttributes<HTMLImageElement>) => {
  return (
    <img
      src="https://raw.githubusercontent.com/Erkhanal/portfolio/master/assets/img/logo.png"
      alt="Logo"
      {...props}
    />
  );
};
