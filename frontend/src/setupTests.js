import '@testing-library/jest-dom';
import { vi } from 'vitest';

global.Element.prototype.scrollIntoView = vi.fn();
global.window.scroll = vi.fn();
global.window.scrollTo = vi.fn();
