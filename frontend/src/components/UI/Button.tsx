import { forwardRef, ReactNode } from 'react'
import { motion } from 'framer-motion'
import { Loader2 } from 'lucide-react'

type ButtonVariant = 'primary' | 'secondary' | 'danger' | 'ghost' | 'outline'
type ButtonSize = 'sm' | 'md' | 'lg'

interface ButtonProps {
  variant?: ButtonVariant
  size?: ButtonSize
  isLoading?: boolean
  leftIcon?: ReactNode
  rightIcon?: ReactNode
  children?: ReactNode
  disabled?: boolean
  className?: string
  onClick?: () => void
}

const variantStyles: Record<ButtonVariant, string> = {
  primary: 'bg-gradient-to-r from-cyber-primary to-cyber-secondary text-black font-semibold hover:shadow-cyber-glow',
  secondary: 'bg-cyber-secondary/20 text-cyber-secondary border border-cyber-secondary/50 hover:bg-cyber-secondary/30',
  danger: 'bg-red-500/20 text-red-400 border border-red-500/50 hover:bg-red-500/30',
  ghost: 'bg-transparent text-gray-400 hover:text-white hover:bg-white/10',
  outline: 'bg-transparent text-cyber-primary border border-cyber-primary/50 hover:bg-cyber-primary/10',
}

const sizeStyles: Record<ButtonSize, string> = {
  sm: 'px-3 py-1.5 text-sm',
  md: 'px-4 py-2 text-base',
  lg: 'px-6 py-3 text-lg',
}

const Button = forwardRef<HTMLButtonElement, ButtonProps>(({
  variant = 'primary',
  size = 'md',
  isLoading = false,
  leftIcon,
  rightIcon,
  children,
  disabled,
  className = '',
  onClick,
}, ref) => {
  return (
    <motion.button
      ref={ref}
      onClick={onClick}
      disabled={disabled || isLoading}
      className={`
        inline-flex items-center justify-center gap-2 rounded-lg
        font-mono transition-all duration-200
        disabled:opacity-50 disabled:cursor-not-allowed
        ${variantStyles[variant]}
        ${sizeStyles[size]}
        ${className}
      `}
      whileHover={!disabled && !isLoading ? { scale: 1.02 } : {}}
      whileTap={!disabled && !isLoading ? { scale: 0.98 } : {}}
    >
      {isLoading ? (
        <Loader2 className="w-4 h-4 animate-spin" />
      ) : leftIcon ? (
        <span>{leftIcon}</span>
      ) : null}
      {children}
      {!isLoading && rightIcon ? <span>{rightIcon}</span> : null}
    </motion.button>
  )
})

Button.displayName = 'Button'

export default Button
